import itertools
import os
import stat

GPUS = [0, 1, 2, 3, 0, 1, 2, 3]
CONF_FILE_NAMES = ["base.yaml", "gaussian_nll.yaml", "mc_dropout.yaml", "quantile_regression.yaml", "der.yaml", "dkl.yaml", "due.yaml", "bnn_elbo.yaml"]
CONF_BASE_DIR = (
    "/p/project/hai_uqmethodbox/nils/uq-method-box/experiments/image-regression/configs/usa_vars_features_extracted"
)
SEEDS = [0]

OOD = True

if __name__ == "__main__":
    for idx, (seed, conf_name) in enumerate(
        itertools.product(SEEDS, CONF_FILE_NAMES)
    ):
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
            command += " default_config=/p/project/hai_uqmethodbox/nils/uq-method-box/experiments/image-regression/configs/usa_vars/default.yaml"
            if OOD:
                command += " experiment.exp_dir=/p/project/hai_uqmethodbox/experiment_output/usa_vars_resnet_ood/"
                command += " wandb.project=usa_vars_resnet_ood"
            else:
                command += " experiment.exp_dir=/p/project/hai_uqmethodbox/experiment_output/usa_vars_resnet/"
                command += " wandb.project=usa_vars_resnet"
        else:
            command += " default_config=/p/project/hai_uqmethodbox/nils/uq-method-box/experiments/image-regression/configs/usa_vars_features_extracted/default.yaml"
            if OOD:
                command += " experiment.exp_dir=/p/project/hai_uqmethodbox/experiment_output/usa_vars_reproduce_ood/"
                command += " wandb.project=usa_vars_reproduce_ood"
            else:
                command += " experiment.exp_dir=/p/project/hai_uqmethodbox/experiment_output/usa_vars_reproduce/"
                command += " wandb.project=usa_vars_reproduce"

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
