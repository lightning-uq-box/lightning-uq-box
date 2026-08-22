(guide)=

# Installation

## pip

The easiest way is to install the package via pip:

```console
pip install lightning-uq-box
```

The above command will install the latest released stable version from [pypi](https://pypi.org/project/lightning-uq-box/).

For the latest development version, you can install directly from the github main branch via:

```console
pip install git+https://github.com/lightning-uq-box/lightning-uq-box.git
```

## conda

The latest stable release can also be installed via [conda](https://docs.conda.io/en/latest/):

```console
conda install lightning-uq-box
```

## spack

Spack was designed for easy installation on High Performance Computing (HPC) environments. See their [documentation](https://spack.readthedocs.io/en/latest/)
for more information.

```console
spack install py-lightning-uq-box
spack load py-lightning-uq-box
```

To install the development version from the latest commit:

```console
spack install py-lightning-uq-box@main
spack load py-lightning-uq-box
```

## from source

To work on the library itself, or to run against the unreleased main branch, clone the repository and let [uv](https://docs.astral.sh/uv/) build the environment from the checked-in lockfile:

```console
git clone https://github.com/lightning-uq-box/lightning-uq-box.git
cd lightning-uq-box
uv sync --all-extras
```

This installs the project in editable mode together with the test, style, and documentation dependencies, pinned to the same versions CI uses. See the {doc}`contribution guide <contribute>` for how to run the tests and linters against it.
