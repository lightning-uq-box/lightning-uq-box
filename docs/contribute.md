# Contribution Guide

We welcome contribution and suggestions to this open-source project. This could be bugs you found, improvements to the documentation or tutorials, as well as new features. The following guide aims to explain the process of contributing to the lightning-uq-box.

## Git

1. Fork the [repository](https://github.com/lightning-uq-box/lightning-uq-box)
2. Clone your fork
3. Create a new branch, make your changes and commit them
    ```console
    $ git checkout main
    $ git checkout -b <branch-name-of-your-changes>
    $ git add <add-all-files-that-you-changed>
    $ git commit -m "commit your changes with a short descriptive message"
    $ git push
    ```
4. Open a Pull Request (PR)

## Tests

Ligthning-UQ-Box uses [Github Actions](https://docs.github.com/en/actions) as a Continous Integration (CI) tool. This means that on every commit there is a set of unit tests that is being executed in order to check that the changes do not break the current version. All unit tests need to pass before the PR can be merged. Additionally, we check code coverage to make see how many lines of code are covered by the unit tests.

For example, if you have implemented a new feature or a new method and want to check the coverage of your unit tests, you can run the following command:

```console
$ pytest --cov=lightning_uq_box/uq_methods --cov-report=term-missing tests/uq_methods/test_changed_method.py
```

## Linters

We use linters to ensure a codebase that follows [PEP-8](https://peps.python.org/pep-0008/) standards.

* [black](https://black.readthedocs.io/) for code formatting
* [isort](https://pycqa.github.io/isort/_ for import ordering
* [flake8](https://flake8.pycqa.org/) for code formatting
* [pydocstyle](https://www.pydocstyle.org/) for docstrings

Black and isort will automatically change your code, while flake8 and pydocstyle will give you warnings.

## Documentation

The documentation is hosted on [Read the Docs](https://readthedocs.org/). 

## Tutorials

Guide for writing tutorials coming.

## UQ Methods

Guide for implementing new methods coming.