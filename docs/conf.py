# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'interpolation_uncertainty'
copyright = '2026, Francis Corpuz, Elias Edideran, Glen Rice'
author = 'Francis Corpuz, Elias Edideran, Glen Rice'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions =  [
    'sphinx.ext.autodoc',    # Pulls docstrings from your Python code
    'sphinx.ext.napoleon',   # Adds support for Google and NumPy docstrings
    'sphinx.ext.viewcode',   # Adds [source] links to generated API pages
    'myst_nb',               # Adds support for Markdown files and Jupyter Notebooks
]

# Prevent Sphinx from re-running your notebooks during make html
nb_execution_mode = "off"

html_theme = 'sphinx_rtd_theme'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_mock_imports = ["gdal", "osgeo"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
