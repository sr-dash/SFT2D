# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------

# Add project source directory to sys.path for autodoc
sys.path.insert(0, os.path.abspath("../sft2d"))  # Adjust if docs/ is one level below root

# -- Project information -----------------------------------------------------

project = 'SFT Simulation 2D and Analysis'
copyright = '2025, Soumyaranjan Dash'
author = 'Soumyaranjan Dash'

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",                # Markdown
    "nbsphinx",                   # Jupyter Notebooks
    "sphinx.ext.autodoc",         # Extracts docstrings
    "sphinx.ext.autosummary",     # Generates summary tables
    "sphinx.ext.napoleon",        # Supports NumPy/Google docstring styles
    "sphinx.ext.viewcode",        # Adds links to highlighted source
    "sphinx_autodoc_typehints",   # Pretty type hints
]

# Markdown + reStructuredText
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Prevent execution of notebooks during build
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# Autosummary + autodoc options
autosummary_generate = True
autodoc_typehints = "description"
autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'logosft2d.jpg'
html_favicon = 'logosft2d.ico'

# Optional: keep the same build date for reproducibility
today_fmt = "%Y-%m-%d"

