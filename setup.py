"""Compatibility shim.

All project metadata, dependencies and packaging configuration live in
``pyproject.toml`` (PEP 621).  This file exists only so that older tooling and
`pip install -e .` on legacy setuptools versions still work; it takes its
configuration entirely from ``pyproject.toml``.
"""

from setuptools import setup

setup()
