# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'arXiv_classification'
copyright = '2024, Aurora Mugnai & Chiara Tocchini'
author = 'Aurora Mugnai & Chiara Tocchini'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 
              'sphinx.ext.napoleon', 
              'sphinx.ext.viewcode',
              'sphinx.ext.autodoc',
              'sphinx.ext.githubpages']

templates_path = ['templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['static']
