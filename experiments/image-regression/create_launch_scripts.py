import argparse
import itertools
import os
import stat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="create_launch_scripts.py", description="Create bash scripts for slurm."
    )

    parser.add_argument(
        "--user", help="Name of user", type=str, required=True, choices=["nina", "nils"]
    )

    args = parser.parse_args()

    GPUS = [0, 1, 2, 3, 0, 1, 2, 3]
    CONF_FILE_NAMES = [
        "base.yaml",
        "gaussian_nll.yaml",
        "mc_dropout.yaml",
        "quantile_regression.yaml",
        "der.yaml",
        "dkl.yaml",
        "bnn_elbo.yaml",
    ]
    CONF_BASE_DIR = f"/p/project/hai_uqmethodbox/{args.user}/uq-method-box/experiments/image-regression/configs/usa_vars_features_extracted"
    SEEDS = [0]

    OOD = True

    ood_type = "gap"

    for idx, (seed, conf_name) in enumerate(itertools.product(SEEDS, CONF_FILE_NAMES)):
        config_file = os.path.join(CONF_BASE_DIR, conf_name)
        if OOD:
            run_command = "python run_usa_vars_ood.py"
        else:
            run_command = "python run_usa_vars_exp.py"
        command = (
            run_command
            + f" config_file={config_file}"
            + f" experiment.seed={seed}"
            + " trainer.devices=[0]"
        )

        base_dir = os.path.basename(CONF_BASE_DIR)
        if base_dir == "usa_vars":
            command += " datamodule.root=/dev/shm/usa_vars/"
            command += f" default_config=/p/project/hai_uqmethodbox/{args.user}/uq-method-box/experiments/image-regression/configs/usa_vars/default.yaml"
            if OOD:
                command += " experiment.exp_dir=/p/project/hai_uqmethodbox/experiment_output/usa_vars_resnet_ood/"
                command += " wandb.project=usa_vars_resnet_ood"
                command += " datamodule._target_=lightning_uq_box.datamodules.USAVarsDataModuleOur"
            else:
                command += " experiment.exp_dir=/p/project/hai_uqmethodbox/experiment_output/usa_vars_resnet/"
                command += " wandb.project=usa_vars_resnet"
                command += " datamodule._target_=lightning_uq_box.datamodules.USAVarsDataModuleOOD"
        else:
            command += f" default_config=/p/project/hai_uqmethodbox/{args.user}/uq-method-box/experiments/image-regression/configs/usa_vars_features_extracted/default.yaml"
            if OOD:
                command += f" experiment.exp_dir=/p/project/hai_uqmethodbox/experiment_output/usa_vars_reproduce_ood_{ood_type}/"
                command += " wandb.project=usa_vars_reproduce_ood"
                command += f" datamodule.ood_type={ood_type}"
                command += " datamodule._target_=lightning_uq_box.datamodules.USAVarsFeatureExtractedDataModuleOOD"
            else:
                command += " experiment.exp_dir=/p/project/hai_uqmethodbox/experiment_output/usa_vars_reproduce/"
                command += " wandb.project=usa_vars_reproduce"
                command += " datamodule._target_=lightning_uq_box.datamodules.USAVarsFeatureExtractedDataModuleOur"

        command = command.strip()

        script = "#!/bin/bash\n"
        script += f"CUDA_VISIBLE_DEVICES={GPUS[idx]}\n"
        script += f"{command}"

        script_path = os.path.join(f"launch_{base_dir}", f"launch_{idx}.sh")
        with open(script_path, "w") as file:
            file.write(script)

        # make bash script exectuable
        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC)

    # create job script

    # launch job script
