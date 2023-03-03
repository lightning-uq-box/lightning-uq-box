## Install dependencies
To work on the project or use this software in your work install the required dependencies as follows:

Move to the root directory, meaning if you run `$ pwd` the printed path should end in `uq_method-box`. 

From here, create a virtual environment:

```
 # create virtual environment
 $ python3 -m venv /path/to/new/virtual/environment/your_environment_name

 # activate venv
 $ source /path/to/new/virtual/environment/your_environment_name/bin/activate/ 

 # install requirements
 $ pip install -r ./requirements.txt
```

This should install dependencies required to run the code in this repo.

## Project Structure
The intended use is to implement all datasets and uq_methods under the [uq_method_box](uq_method_box/) and all code unique to our experiments that we want to run in [experiments](experiments/). This is a brief description of the directory structure:

- [datasets](uq_method_box/datasets/): Pytorch Dataset classes that define how to return a sample from the dataset.
- [datamodules](uq_method_box/datamodules/): PytorchLightning [Datamodules](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html) that define training, validation and test set corresponding to the implemented datasets.
- [eval_utils](uq_method_box/eval_utils/): Utility functions and scripts to evaluate predictive uncertainty
- [models](uq_method_box/models/): If we require any unique implementation of models that are needed for experiments or are helpful for tutorials (like the notebook MLP implementation), we can define them here (separate files)
- [train_utils](uq_method_box/train_utils/): Training utility function and scripts should make it easy for us and users to train models with UQ-methods. 
- [viz_utils](uq_method_box/train_utils/): Visualization utility functions and scripts that are useful for tutorials or evaluation of experiment results
- [uq_methods](uq_methods_box/uq_methods): Here each UQ-Method should be implemented as its own class (and python file) in the form of a PytorchLightning [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html). If possible it should inherit from the base class to minimize code duplication. The methods should be "model-agnostic" in the sense that they should work with any backbone like ResNet etc. and support large datasets if applicable.

## Run UCI Experiments
To run UCI experiments navigate to the root directory of the project in your command line. In [configs](experiments/configs/) are examples of config files that let you 
configure the kind of experiment you want to execute. To launch an experiment:

```
   $ python run_uci_experiments.py --config_path "./experiments/configs/name_of_config_you_want.yaml"
```

The directory structure for the results that your experiment produces looks something like that, where per seed you have M ensemble members:

```
├── uci_base_model_03-02-2023_10:52:07
│   └── seed_0
│       ├── model_0
│       │   ├── csv_logs
│       │   │   └── version_0
│       │   │       ├── hparams.yaml
│       │   │       └── metrics.csv  # metrics during training
│       │   ├── epoch=9-step=30.ckpt # saved lightining checkpoint
│       │   └──run_config.yaml
│       │   
│       ├── model_1
│       │   ├── csv_logs
│       │   │   └── version_0
│       │   │       ├── hparams.yaml
│       │   │       └── metrics.csv
│       │   ├── epoch=9-step=30.ckpt
│       │   └── run_config.yaml
│       ├── model_2
│       │   ├── csv_logs
│       │   │   └── version_0
│       │   │       ├── hparams.yaml
│       │   │       └── metrics.csv
│       │   ├── epoch=9-step=30.ckpt
│       │   └── run_config.yaml
│       └── prediction
│           ├── csv_logs
│           │   └── version_0
│           │       └── hparams.yaml
│           └── predictions.csv # stores all predictions and model outputs on test set
```

## Evaluation setup
There exists some utility functions that iterate over the experiment collection directory (so the directory where all the individual experiments are stored) in `experiments/eval_utils.py`. This script can
be executed with the python script `compute_results_uci.py` from the project root directory (same place you also execute `run_uci_experiments.py`). This will generate a csv file in `experiments/experiments/` with the name `results.csv`.
This csv file collects all the indiviual results from the `predictions.csv` in each seed directory into a big table, based on which we can write scripts for visualizations or automatic table generation for the paper. As a minimal example, I included
a `explore_results.ipynb` notebook in the `experiments/experiments` directory. We should check some papers across the community for some great visualizations where many models are evaluated across many datasets.

## TODOs
I think the UCI-Experiments are a great opportunity to both check our implementations of methods, but also be a "rehearsal" for the EO data experiments that we want to run. Therefore, we should invest some additional thoughts in criticizing the existing workflow of experiments so that we can fix problems, improve it, and make our lives easier for the EO experiments. This could be anything like
- the experiment directory structure is silly, because we have to do XYZ in the evaluation, and ABC would make it easier
- we should track some additional variables that make it easier or more convenient to evaluate experiments
- the naming in the config files is confusing and it should rather be this to make it clearer
- etc.

The goal is to be "as lazy as possible" in the effort it takes to execute experiments but also have confidence in the worklfow, so anything fishy you see make a note of it and we can look into it and hopefully fix it :) 

