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
Before running experiments for the first time, create the folder experiments in the folder experiments, i.e. \eperiments\experiments. Also create a folder data in experiments, \experiments\data.


To run UCI experiments navigate to the root directory of the project in your command line. In [configs](experiments/configs/) are examples of config files that let you configure the kind of experiment you want to execute. To launch an experiment, change the directory under root in the config file:

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

Before evaluating experiments for the first time, before create the folder exp_results \experiments\experiments\exp_results.

There exists some utility functions that iterate over the experiment collection directory (so the directory where all the individual experiments are stored) in `experiments/eval_utils.py`. This script can
be executed with the python script `compute_results_uci.py` from the project root directory (same place you also execute `run_uci_experiments.py`). This will generate a csv file in `experiments/experiments/` with the name `results.csv`.
This csv file collects all the indiviual results from the `predictions.csv` in each seed directory into a big table, based on which we can write scripts for visualizations or automatic table generation for the paper. As a minimal example, I included
a `explore_results.ipynb` notebook in the `experiments/experiments` directory. We should check some papers across the community for some great visualizations where many models are evaluated across many datasets.

```
    $ python compute_results_uci.py --exp_dir_path "./experiments/experiments/"
    #with new code
    $ python compute_results_uci.py --exp_dir_path ".\experiments\experiments" --save_path ".\experiments\experiments\exp_results"
```

## TODOs
I think the UCI-Experiments are a great opportunity to both check our implementations of methods, but also be a "rehearsal" for the EO data experiments that we want to run. Therefore, we should invest some additional thoughts in criticizing the existing workflow of experiments so that we can fix problems, improve it, and make our lives easier for the EO experiments. This could be anything like
- the experiment directory structure is silly, because we have to do XYZ in the evaluation, and ABC would make it easier
- we should track some additional variables that make it easier or more convenient to evaluate experiments
- the naming in the config files is confusing and it should rather be this to make it clearer
- etc.

The goal is to be "as lazy as possible" in the effort it takes to execute experiments but also have confidence in the worklfow, so anything fishy you see make a note of it and we can look into it and hopefully fix it :) 


## Conda environment

I have run the commands in this order to create a conda environment with which I am able to run experiments. Mabye the same order will work on Windows as well. If conda install does not work, try using pip install instead.

```
    $  conda create --name ninaEnv python=3.9
    $  conda activate ninaEnv
    $  conda install pytorch torchvision cpuonly -c pytorch
    $  conda install -c conda-forge lightning
    $  conda install -c conda-forge torchgeo
    $  conda install -c conda-forge pandas
    $  conda install -c conda-forge laplace-torch
    $  conda install -c conda-forge ruamel.yaml
    $  conda install -c conda-forge ipykernel
    $  conda install -c conda-forge seaborn
    $  pip install uncertainty-toolbox
```

## Code Reviews 
All updates to the code base should go through a review process. We use GitHub Pull Requests for this. You can head over to
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Linters

From [Torchgeo](https://github.com/microsoft/torchgeo/blob/main/docs/user/contributing.rst)

In order to remain [PEP-8](https://peps.python.org/pep-0008/) compliant and maintain a high-quality codebase, we use several linting tools:

- [black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import ordering
- [flake8](https://flake8.pycqa.org/) for code formatting
- [pydocstyle](https://www.pydocstyle.org/) for docstrings

All of these tools should be used from the root of the project to ensure that our configuration files are found. Black, and isort are relatively easy to use, and will automatically format your code for you:

```
   $ black .
   $ isort .
```

Flake8 and pydocstyle won't format your code for you, but they will warn you about potential issues with your code or docstrings:

```
   $ flake8
   $ pydocstyle
```

You can also use [git pre-commit hooks](https://pre-commit.com/) to automatically run these checks before each commit. pre-commit is a tool that automatically runs linters locally, so that you don't have to remember to run them manually and then have your code flagged by CI. You can setup pre-commit with:

```
   $ pip install pre-commit
   $ pre-commit install
   $ pre-commit run --all-files
```

Now, every time you run ``git commit``, pre-commit will run and let you know if any of the files that you changed fail the linters. If pre-commit passes then your code should be ready (style-wise) for a pull request. Note that you will need to run ``pre-commit run --all-files`` if any of the hooks in ``.pre-commit-config.yaml`` change, see [here](https://pre-commit.com/#4-optional-run-against-all-the-files>).

### Tests
We will have to write unit tests to make sure our code runs as expected. This is good practice to check our own working and a necessity before open-sourcing the repo. Tests should follow the exact directory structure of the `uq_method_box` directory for each individual python file with the name `test_[filename].py`.

### Documentation
We aim to provide extensive documentation. All the docstrings written inside the python files are generated into a readable documentation with the help of [Sphinx](https://www.sphinx-doc.org/en/master/) and [ReadTheDocs](https://readthedocs.org/). This means that all functions and classes should be clearly documented. We can extend the documentation by writing tutorials or additional information in the corresponding .rst file in the [api](docs/api/). In order, to preview the documentation locally you can do the following:


Make sure the dependencies are installed inside your virtual environment:

```
   $ pip install .[docs]
   $ cd docs
   $ pip install -r requirements/requirements.txt
```


Then run the following commands:

```
   $ make clean
   $ make html
```

The resulting HTML files can be found in ``_build/html``. Open ``index.html`` in your browser to navigate the project documentation. If you fix something, make sure to run ``make clean`` before running ``make html`` or Sphinx won't rebuild all of the documentation.
