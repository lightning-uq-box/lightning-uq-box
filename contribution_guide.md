## Install dependencies
To work on the project or use this software in your work install the required dependencies as follows:

Move to the root directory, meaning if you run `$ pwd` the printed path should end in `uq_method-box`. We require python 3.9 or higher.

I encountered some issues with packages when using conda, however, pip works. So either in a conda
or venv environment run the following installs.

## Conda environment

```
   $ conda create --name myEnvName python=3.9  
   $ pip install git+https://github.com/microsoft/torchgeo.git
   $ pip install pandas
   $ ip install bayesian-torch
   $ pip install laplace-torch
   $ pip install uncertainty-toolbox
   $ pip install ruamel.yaml
   $ pip install ipykernel
   $ pip install seaborn 
```
This should install dependencies required to run the code in this repo.

Additionally, install the following for code formatting and pytests as explained in the section further down below.

```
   $ pip install black
   $ pip install isort
   $ pip install flake8
   $ pip install pytest
   $ pip install pytest-cov
```

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

At the moment we are using raw config files, but I am planning to adopt the framework of [hydra](https://github.com/facebookresearch/hydra) for better control and reproducibility, as you can target class instances better.

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
We are using unit tests to make sure our code runs as expected. This is merely checking the "mechanics", for example that our implementations work with trainers or in Toy Examples. It does not give you any guarantees about the "correctness" of the implementation. Tests should follow the exact directory structure of the `uq_method_box` directory for each individual python file with the name `test_[filename].py`. Later this tests will also be run automatically on every commit and will be required to pass before a PR can be merged to the main branch. We use [pytest](https://docs.pytest.org/en/7.2.x/) for the implementation of unit tests. Additionally, we also use [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) which tests the coverage of our unit tests meaning which written lines of code are actually tested or covered by our tests.

To test your code install the following two libraries in your environment:

```
   $ pip install pytest
   $ pip install pytest-cov
```

If you implement a new method for example which is placed in `uq_method_box/uq_methods/my_new_method.py`, there should be a corresponding file in `tests/uq_methods/test_my_new_method.py` which implements unit tests for your implementation. To execute the tests use the following command:

```
   $ pytest --cov=uq_method_box/uq_methods/ --cov-report=term-missing tests/uq_methods/test_my_new_method.py
```

The command above only executes the tests in `test_my_new_method.py` but gives you a coverage report of all files contained in `uq_method_box/uq_methods/`. The below output is an example of running the tests on all files under `uq_method_box/uq_methods/`:

```
   $ pytest --cov=uq_method_box/uq_methods --cov-report=term-missing tests/uq_methods/
   platform linux -- Python 3.9.12, pytest-7.2.2, pluggy-1.0.0
   plugins: anyio-3.6.2, hydra-core-1.3.1, cov-4.0.0
   collected 32 items                                                                                                                                                                                                                                            

   tests/uq_methods/test_base.py ....                                                 [ 12%]
   tests/uq_methods/test_cqr_model.py ...                                             [ 21%]
   tests/uq_methods/test_deep_ensemble_model.py ....                                  [ 34%]
   tests/uq_methods/test_deep_evidential_regression.py ...                            [ 43%]
   tests/uq_methods/test_deterministic_gaussian.py ..                                 [ 50%]
   tests/uq_methods/test_laplace_model.py ...                                         [ 59%]
   tests/uq_methods/test_mc_dropout_model.py ......                                   [ 78%]
   tests/uq_methods/test_quantile_regression_model.py ...                             [ 87%]
   tests/uq_methods/test_utils.py ....                                                [100%]

   ---------- coverage: platform linux, python 3.9.12-final-0 -----------
   Name                                                     Stmts   Miss  Cover   Missing
   --------------------------------------------------------------------------------------
   uq_method_box/uq_methods/__init__.py                         9      0   100%
   uq_method_box/uq_methods/base.py                            64      0   100%
   uq_method_box/uq_methods/cqr_model.py                       58      0   100%
   uq_method_box/uq_methods/deep_ensemble_model.py             43      0   100%
   uq_method_box/uq_methods/deep_evidential_regression.py      46      0   100%
   uq_method_box/uq_methods/deterministic_gaussian.py          24      0   100%
   uq_method_box/uq_methods/laplace_model.py                   61      0   100%
   uq_method_box/uq_methods/mc_dropout_model.py                36      0   100%
   uq_method_box/uq_methods/quantile_regression_model.py       23      0   100%
   uq_method_box/uq_methods/utils.py                           28      0   100%
   --------------------------------------------------------------------------------------
   TOTAL                                                      392      0   100%


   ================================== 32 passed in 9.74s =======================================
```

If certain lines are not coverd by the unit tests, they will be explicitly listed in the "Missing" column any you can adjust your tests accordingly (or remove bugs or unecessary code as needed).

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
