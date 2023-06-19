"""Create bash and job script to run experiment."""
import itertools
import os
import stat

GPUS = [2, 3]
CONF_FILE_NAMES = ["gaussian_nll.yaml"]
CONF_BASE_DIR = (
    "/mnt/SSD2/nils/uq-method-box/experiments/image-regression/configs/usa_vars"
)
SEEDS = [0]

if __name__ == "__main__":
    for idx, (gpu, seed, conf_name) in enumerate(
        itertools.product(GPUS, SEEDS, CONF_FILE_NAMES)
    ):
        config_file = os.path.join(CONF_BASE_DIR, conf_name)

        command = (
            "python run_usa_vars_exp.py"
            + f" config_file={config_file}"
            + f" experiment.seed={seed}"
            + " trainer.devices=[0]"
        )
        command = command.strip()

        script = "#!/bin/bash\n"
        script += f"CUDA_VISIBLE_DEVICES={gpu}\n"
        script += f"{command}"

        script_path = f"launch_{idx}.sh"
        with open(script_path, "w") as file:
            file.write(script)

        # make bash script exectuable
        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC)

    # create job script

    # launch job script
