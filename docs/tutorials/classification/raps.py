import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from lightning import Trainer
from lightning.pytorch import seed_everything
from orig_conformal import ConformalModel
from orig_utils import (
    AverageMeter,
    compute_accuracy,
    coverage_size,
    get_model,
    validate,
)
from tqdm import tqdm

from lightning_uq_box.uq_methods import RAPS

cudnn.benchmark = True
import pandas as pd

plt.rcParams["figure.figsize"] = [14, 5]


def compute_empirical_coverage(S: list[Sequence[int]], y: list[int]) -> float:
    """Compute the empirical coverage of the predictions.

    Args:
        S: List of sets of predicted labels for each instance.
        y: List of true labels for each instance.
    """
    coverage = np.mean([y[i].item() in S[i] for i in range(len(y))])
    return coverage

my_temp_dir = "."

from torch.utils.data import DataLoader, random_split

# Imagenet datamodule
from torchvision import datasets, transforms


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return {"image": image, "label": label}
        # return image, label


def collate_fn(batch):
    """turn dictionary output to tuple output."""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return images, labels


# Define the transform
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the dataset
dataset = CustomImageFolder("imagenet_val", transform=transform)


# Split the dataset into validation and calibration sets
torch.manual_seed(0)
val_size = int(0.8 * len(dataset))
cal_size = len(dataset) - val_size
val_dataset, cal_dataset = random_split(dataset, [val_size, cal_size])

# Create the DataLoaders
batch_size = 512
num_workers = 24
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)
cal_dataloader = DataLoader(
    cal_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

their_cal_loader = DataLoader(
    cal_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    collate_fn=collate_fn,
)
their_val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    collate_fn=collate_fn,
)


# THE ORIGINAL RAPS IMPLEMENTATION
def fit_original_raps(model: torch.nn.Module):
    # allow sets of size zero
    allow_zero_sets = False
    # use the randomized version of conformal
    randomized = True

    # Conformalize model
    model = ConformalModel(
        model,
        their_cal_loader,
        alpha=0.1,
        lamda=0,
        randomized=randomized,
        allow_zero_sets=allow_zero_sets,
    )

    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    validation_metrics = validate(their_val_loader, model, print_bool=True)

    return model, {
        "top1": validation_metrics[0],
        "top5": validation_metrics[1],
        "coverage": validation_metrics[2],
        "size": validation_metrics[3],
        "temperature": model.T.detach().item(),
    }


def fit_my_raps(orig_model: torch.nn.Module):
    raps = RAPS(orig_model.model, lamda_param=orig_model.lamda, kreg=orig_model.kreg, randomized=True, allow_zero_sets=False)
    raps.input_key = "image"
    raps.target_key = "label"

    raps_trainer = Trainer(
        default_root_dir=my_temp_dir,
        accelerator="gpu",
        devices=[0],
        inference_mode=False,
    )
    # need to pass the calibration loader here because that is the dataset over which Q_hat is computed
    raps_trainer.validate(raps, dataloaders=cal_dataloader)

    # now adjust the logits
    raps = raps.to(device)
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    coverage = AverageMeter("coverage")
    size = AverageMeter("size")
    # evaluate on full validation set
    for batch in tqdm(val_dataloader):
        image = batch["image"].to(device)
        with torch.no_grad():
            out = raps.predict_step(image)
            output, S = out["pred"], out["pred_set"]
            S = [s.cpu().numpy() for s in S]

            # measure accuracy and record loss
            prec1, prec5 = compute_accuracy(
                output.cpu(), batch["label"].cpu(), topk=(1, 5)
            )
            cvg, sz = coverage_size(S, batch["label"])

            # Update meters
            top1.update(prec1.item() / 100.0, n=output.shape[0])
            top5.update(prec5.item() / 100.0, n=output.shape[0])
            coverage.update(cvg, n=output.shape[0])
            size.update(sz, n=output.shape[0])

    return raps, {
        "top1": top1.avg,
        "top5": top5.avg,
        "coverage": coverage.avg,
        "size": size.avg,
        "temperature": raps.temperature.detach().item(),
    }

def run_seed(seed, device):
    print(f"RUNNING SEED {seed} ON {device}")
    seed_everything(seed)
    result_dict = {}
    model_names = ["ResNet18", "ResNet50", "ResNet101", "ResNet152"]

    torch.set_float32_matmul_precision("medium")

    for name in model_names:
        print("now running model",name)
        result_dict[name] = {}
        model = get_model(name)

        # set device
        device = torch.device(device)
        model = model.to(device)

        conformal_model, orig_metrics = fit_original_raps(model)
        print(orig_metrics)
        raps, metrics = fit_my_raps(conformal_model)
        result_dict[name]["orig"] = orig_metrics
        result_dict[name]["my"] = metrics

    df = pd.DataFrame(result_dict)
    df = df.stack()
    df = df.apply(pd.Series)
    df = df.swaplevel(0, 1).sort_index()
    df.reset_index(inplace=True)
    df.rename(columns={"level_0": "model", "level_1": "version"}, inplace=True)
    df.to_csv(os.path.join("results", f"imagenet_results_{seed}.csv"), index=False)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_seeds = 5
    for i in tqdm(range(num_seeds)):
        run_seed(i, device)