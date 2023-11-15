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
    "myst_parser",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
master_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
templates_path = ["_templates"]

source_dirs = ["api", "tutorials"]

# General information about the project.
project = "Lightning-UQ-Box"
# copyright = "."
version = lightning_uq_box.__version__
release = lightning_uq_box.__version__

exclude_patterns = ["_build"]
html_theme = "sphinx_book_theme"  # "sphinx_book_theme"
html_title = "Lightning-UQ-Box"
# html_logo = "_static/zap.svg"
# html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/nilsleh/lightning-uq-box",
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

# # sphinx.ext.intersphinx
# # intersphinx_mapping = {
# #     "kornia": ("https://kornia.readthedocs.io/en/stable/", None),
# #     "matplotlib": ("https://matplotlib.org/stable/", None),
# #     "numpy": ("https://numpy.org/doc/stable/", None),
# #     "python": ("https://docs.python.org/3", None),
# #     "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
# #     "pyvista": ("https://docs.pyvista.org/version/stable/", None),
# #     "rasterio": ("https://rasterio.readthedocs.io/en/stable/", None),
# #     "rtree": ("https://rtree.readthedocs.io/en/stable/", None),
# #     "segmentation_models_pytorch": ("https://smp.readthedocs.io/en/stable/", None),
# #     "sklearn": ("https://scikit-learn.org/stable/", None),
# #     "timm": ("https://huggingface.co/docs/timm/main/en/", None),
# #     "torch": ("https://pytorch.org/docs/stable", None),
# #     "torchvision": ("https://pytorch.org/vision/stable", None),
# # }
