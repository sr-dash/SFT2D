# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the repo root so 'sft2d' is importable
sys.path.insert(0, os.path.abspath("..")) 

# -- Project information -----------------------------------------------------
project = 'SFT Simulation 2D and Analysis'
author = 'Soumyaranjan Dash'
copyright = '2025, Soumyaranjan Dash'

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",                # Markdown support
    "nbsphinx",                   # Notebook support
    "sphinx.ext.autodoc",         # API docs from docstrings
    "sphinx.ext.autosummary",     # Generate summary tables
    "sphinx.ext.napoleon",        # Google/NumPy style docstrings
    "sphinx.ext.viewcode",        # Link to source code
    "sphinx_autodoc_typehints",   # Type hints in docs
]

# Generate autosummary pages automatically
autosummary_generate = True
autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# nbsphinx options
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# -- HTML output options -----------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'logosft2d.jpg'
html_favicon = 'logosft2d.ico'

