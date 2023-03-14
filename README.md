# uq-method-box

A set of datasets and models to evaluate predictive uncertainty on regression tasks. As a PyTorch library it relies heavily on PytorchLightning and GPytorch.
The goal of this library is threefold:

1. Provde access to a variety of Uncertainty Quantification methods for Modern Deep Neural Networks that work with a range of models.
2. Make it easy to compare methods on a new dataset or a new method on existing datasets.
3. Focus on reproducibility of experiments with miminum boiler plate code and standardized evaluation protocols.

## Documentation 
We aim to provide an extensive documentation on all included UQ-methods to clearly state assumptions and pitfalls, as well as tutorials that illustrate these. 

## Experiments
The experiments conducted for our paper "Title here." are contained in the [experiments](experiments/) directory. To reproduce the results in our paper follow these instructions.

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