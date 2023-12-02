import torch
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from lightning import Trainer
from lightning.pytorch import seed_everything

from lightning_uq_box.uq_methods import RAPS
from tqdm import tqdm

from orig_utils import compute_accuracy, coverage_size, get_model, validate


from orig_utils import validate
from orig_conformal import ConformalModel

from orig_utils import AverageMeter, coverage_size, compute_accuracy
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

plt.rcParams["figure.figsize"] = [14, 5]


torch.set_float32_matmul_precision("medium")


def compute_empirical_coverage(S: list[Sequence[int]], y: list[int]) -> float:
    """Compute the empirical coverage of the predictions.

    Args:
        S: List of sets of predicted labels for each instance.
        y: List of true labels for each instance.
    """
    coverage = np.mean([y[i].item() in S[i] for i in range(len(y))])
    return coverage


seed_everything(0)
my_temp_dir = "."

# Imagenet datamodule
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


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
batch_size = 128
num_workers = 8
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)
cal_dataloader = DataLoader(
    cal_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

their_cal_loader = DataLoader(
    cal_dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=False,
    collate_fn=collate_fn,
)
their_val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=False,
    collate_fn=collate_fn,
)
print(len(val_dataloader), len(cal_dataloader))
# y_true, y_pred = [], []
# for batch in tqdm(their_cal_loader):
#     image = batch[0].to(device)
#     # label = batch["label"].to(device)
#     with torch.no_grad():
#         y_true.append(batch[1])
#         y_pred.append(timm_model(image).cpu())

# y_true = torch.cat(y_true)
# y_pred = torch.cat(y_pred)

# print("DEFAULT MODEL")
# # Compute accuracy
# accuracy = (torch.argmax(y_pred, dim=1) == y_true).float().mean()
# print(accuracy)
# # compute coverage
# S_pred = [set(torch.topk(y_pred[i], 2).indices.tolist()) for i in range(y_pred.shape[0])]
# coverage = compute_empirical_coverage(S_pred, y_true)
# print(coverage)


# ## Model
# Specify the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# timm_model = timm.create_model('resnet18', pretrained=True)
# timm_model = timm_model.to(device)
import torchvision

timm_model = torchvision.models.resnet18(pretrained=True, progress=True).cuda()


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
    validation_metrics = validate(their_cal_loader, model, print_bool=True)

    return model, {
        "top1": validation_metrics[0],
        "top5": validation_metrics[1],
        "coverage": validation_metrics[2],
        "size": validation_metrics[3],
        "temperature": model.T.detach().item(),
    }


def fit_my_raps(orig_model: torch.nn.Module):
    raps = RAPS(orig_model.model, lamda_param=orig_model.lamda, kreg=orig_model.kreg)
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
    raps.temperature = orig_model.T

    # now adjust the logits
    raps = raps.to(device)
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    coverage = AverageMeter("RAPS coverage")
    size = AverageMeter("RAPS size")
    for batch in tqdm(cal_dataloader):
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
        "temperature": raps.temperature.detach().item()
    }


result_dict = {}
model_names = ["ResNet18", "ResNet50", "ResNet101", "ResNet152"]

for name in model_names:
    result_dict[name] = {}
    model = get_model(name)

    conformal_model, orig_metrics = fit_original_raps(model)

    raps, metrics = fit_my_raps(conformal_model)
    result_dict[name]["orig"] = orig_metrics
    result_dict[name]["my"] = metrics

print(result_dict)

import pandas as pd
df = pd.DataFrame(result_dict)
df = df.stack()
df = df.apply(pd.Series)
df = df.swaplevel(0, 1).sort_index()

print(df)
import pdb
pdb.set_trace()
print(0)
