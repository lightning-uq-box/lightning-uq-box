# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import pytorch_sphinx_theme

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))

import uq_method_box  # noqa: E402

project = "uq-regression-box"
copyright = "2023, Nils Lehmann"
author = uq_method_box.__author__
release = uq_method_box.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# source file parsers
source_parsers = {".rst": "restructuredtext", ".ipynb": "sphinxcontrib.jupyter"}

# source file suffixes
source_suffix = [".rst"]

# The source directory containing Sphinx source files.
source_dirs = ["api", "tutorials"]

# Sphinx 3.0+ required for:
# autodoc_typehints_description_target = "documented"
needs_sphinx = "4.0"

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_kernel_name = "python"

nbsphinx_prolog = """
{% set host = "https://colab.research.google.com" %}
{% set repo = "microsoft/torchgeo" %}
{% set urlpath = "docs/" ~ env.docname ~ ".ipynb" %}
{% if "dev" in env.config.release %}
    {% set branch = "main" %}
{% else %}
    {% set branch = "releases/v" ~ env.config.version %}
{% endif %}

.. image:: {{ host }}/assets/colab-badge.svg
   :class: colabbadge
   :alt: Open in Colab
   :target: {{ host }}/github/{{ repo }}/blob/{{ branch }}/{{ urlpath }}
"""


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "furo"
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# css adjustments
html_static_path = ["_static"]
html_css_files = ["button-width.css", "notebook-prompt.css"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "pytorch_project": "docs",
    "navigation_with_keys": True,
    "analytics_id": "UA-209075005-1",
}

# sphinx.ext.intersphinx
intersphinx_mapping = {
    "kornia": ("https://kornia.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "pyvista": ("https://docs.pyvista.org/version/stable/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/stable/", None),
    "rtree": ("https://rtree.readthedocs.io/en/stable/", None),
    "segmentation_models_pytorch": ("https://smp.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "timm": ("https://huggingface.co/docs/timm/main/en/", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "torchvision": ("https://pytorch.org/vision/stable", None),
}

# -- Extension configuration -------------------------------------------------

# sphinx.ext.autodoc
autodoc_default_options = {
    "members": True,
    "special-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
