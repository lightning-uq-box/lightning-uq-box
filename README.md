# uq-method-box

A set of datasets and models to evaluate predictive uncertainty on regression tasks. As a PyTorch library it relies heavily on PytorchLightning and GPytorch.
The goal of this library is threefold:

1. Provde access to a variety of Uncertainty Quantification methods for Modern Deep Neural Networks that work with a range of models.
2. Make it easy to compare methods on a new dataset or a new method on existing datasets.
3. Focus on reproducibility of experiments with miminum boiler plate code and standardized evaluation protocols.

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


## Documentation 
We aim to provide an extensive documentation on all included UQ-methods to clearly state assumptions and pitfalls, as well as tutorials that illustrate these.

## Experiments
The experiments conducted for our paper "Title here." are contained in the [experiments](experiments/) directory. To reproduce the results in our paper follow these instructions.

## Collaborators
The intended use is to implement all datasets and uq_methods under the [uq_method_box](uq_method_box/) and all code unique to our experiments that we want to run in [experiments](experiments/). This is a brief description of the directory structure:

- [datasets](uq_method_box/datasets/): Pytorch Dataset classes that define how to return a sample from the dataset.
- [datamodules](uq_method_box/datamodules/): PytorchLightning [Datamodules](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html) that define training, validation and test set corresponding to the implemented datasets.
- [eval_utils](uq_method_box/eval_utils/): Utility functions and scripts to evaluate predictive uncertainty
- [models](uq_method_box/models/): If we require any unique implementation of models that are needed for experiments or are helpful for tutorials (like the notebook MLP implementation), we can define them here (separate files)
- [train_utils](uq_method_box/train_utils/): Training utility function and scripts should make it easy for us and users to train models with UQ-methods. 
- [viz_utils](uq_method_box/train_utils/): Visualization utility functions and scripts that are useful for tutorials or evaluation of experiment results
- [uq_methods](uq_methods_box/uq_methods): Here each UQ-Method should be implemented as its own class (and python file) in the form of a PytorchLightning [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html). If possible it should inherit from the base class to minimize code duplication. The methods should be "model-agnostic" in the sense that they should work with any backbone like ResNet etc. and support large datasets if applicable.


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

### Tests
We will have to write unit tests to make sure our code runs as expected. This is good practice to check our own working and a necessity before open-sourcing the repo. Tests should follow the exact directory structure of the `uq_method_box` directory for each individual python file with the name `test_[filename].py`.


## Citation
If you use this software in your work, please cite...