# Contribution Guide

We welcome contributions and suggestions to this open-source project. This could be bugs you found, improvements to the documentation or tutorials, as well as new features and methods. The following guide aims to explain the process of contributing to the lightning-uq-box.

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

Lightning-UQ-Box uses [Github Actions](https://docs.github.com/en/actions) as a Continuous Integration (CI) tool. This means that on every commit there is a set of unit tests that is being executed in order to check that the changes do not break the current version. All unit tests need to pass before the PR can be merged. Additionally, we check code coverage to make see how many lines of code are covered by the unit tests.

For example, if you have implemented a new feature or a new method and want to check the coverage of your unit tests, you can run the following command:

```console
$ pytest --cov=lightning_uq_box/uq_methods --cov-report=term-missing tests/uq_methods/test_changed_method.py
```

## Linters

We use a linter to ensure a codebase that follows [PEP-8](https://peps.python.org/pep-0008/) standards.

* `ruff <https://docs.astral.sh/ruff/>`_ for code formatting

You can apply these tools from the project root where the necessary configuration files are found. To test with ruff, you can do:

```console
$ ruff check
$ ruff format
```

These tools should be used from the root of the project to ensure that our configuration files are found. Ruff is relatively easy to use, and will automatically fix most issues it encounters:

You can also use `git pre-commit hooks <https://pre-commit.com/>`_ to automatically apply these checks before each commit. You can use pre-commit as follows:

```console
$ pip install pre-commit
$ pre-commit install
$ pre-commit run --all-files
```

## Documentation

The documentation is hosted on [Read the Docs](https://readthedocs.org/). If you are making changes to the documentation, it can be useful to inspect the changes locally before committing them. You can follow these steps:

1. Move to the `docs` directory
2. In the `conf.py` file look at the very last line and uncomment if you want to speed up the documentation build. This will not execute the notebooks and just build the rest of the documentation. However, when you are making changes to the notebooks as well, you should leave it uncommented as the notebooks won't be updated with your changes otherwise.
3. Run `make clean` followed by `make html`
4. Once that command finishes, there will be a `index.html` file under `docs/_build/html`. Paste the full path to that file into your web browser to inspect what the documentation would look like with your changes

## Tutorials

We aim to give comprehensive tutorials that illustrate different UQ-Methods. If there are specific use cases you would like to see covered (from whatever domain) please do not hesitate to reach out. The majority of our tuturials utilize toy datasets because they can be used to visualize model predictions easily and see the behavior of different UQ Methods. However, we would also love to support more involved tutorials that showcase the use of UQ in a variety of domains. If you have any ideas or suggestions for tutorials or are looking for help in setting up a tutorial, we would love to help.
