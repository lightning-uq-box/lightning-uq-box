# # Configuration file for the Sphinx documentation builder.
# #
# # For the full list of built-in configuration values, see the documentation:
# # https://www.sphinx-doc.org/en/master/usage/configuration.html

# # -- Project information -----------------------------------------------------
# # https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))

import lightning_uq_box

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_togglebutton",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
master_doc = "index"
# list of source suffix to include .py for jupytext
source_suffix = [".rst", ".md", ".py"]
# source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
templates_path = ["_templates"]

# this is needed for jupytext
nbsphinx_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:light"}]}

source_dirs = ["api", "tutorials"]

# General information about the project.
project = "Lightning-UQ-Box"
copyright = "2023, lightning-uq-box"
version = lightning_uq_box.__version__
release = lightning_uq_box.__version__

# exclude ipynb for jupytext
exclude_patterns = ["_build", "**/*.ipynb", "earth_observation/*.ipynb"]
# exclude_patterns = ["_build"]
html_theme = "sphinx_book_theme"
html_title = "Lightning-UQ-Box"
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/lightning-uq-box/lightning-uq-box",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
        "colab_url": "https://colab.research.google.com/",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}


autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
# sphinx.ext.intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}
