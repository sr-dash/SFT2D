# Contributing to SFT2D

Welcome, and thank you for your interest in contributing to **SFT2D: Simulating magnetic flux transport processes on solar/stellar surfaces!**
This document explains how to contribute, what we expect from contributors, and how to keep the repository organized and welcoming.

---

## 📜 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)

   * [Filing Issues](#filing-issues)
   * [Suggesting Enhancements](#suggesting-enhancements)
   * [Pull Requests](#pull-requests)
   * [Branching & Commits](#branching--commits)
3. [Development Setup](#development-setup)
4. [Coding Standards & Style](#coding-standards--style)
5. [Testing & Quality Assurance](#testing--quality-assurance)
6. [Documentation](#documentation)
7. [Releases & Versioning](#releases--versioning)
8. [Licensing](#licensing)
9. [Acknowledgements](#acknowledgements)
10. [Contact](#contact)

---

## 🧭 Code of Conduct

Please review the repository’s `CODE_OF_CONDUCT.md`.
By participating (filing issues, submitting pull requests, commenting, etc.), you agree to abide by its terms.
We aim to maintain an inclusive, respectful, and collaborative community.

---

## 🚀 How to Contribute

### Filing Issues

If you find a bug, have a feature idea, or notice something unclear, please [open an issue](../../issues).
Include as much detail as possible:

* ✅ A descriptive title and summary
* 🪲 What you expected vs. what happened
* 🧩 Steps to reproduce (if applicable)
* 💻 Your environment (Python version, OS, `sft2d` version)
* 📎 Logs, screenshots, or minimal reproducible examples

### Suggesting Enhancements

Feature requests are welcome! When suggesting one, please include:

* A clear description of the current and desired behavior
* Why the change is beneficial (scientific motivation, usability, etc.)
* Optional: how you imagine it could be implemented

### Pull Requests

We encourage pull requests (PRs) for bug fixes, features, and documentation improvements.

**Typical workflow:**

1. Fork the repository
2. Create a new branch from `main`:

   ```bash
   git checkout -b feature/<short-name>
   ```
3. Make your changes and include tests/docs if needed
4. Commit your work with a clear message
5. Push to your fork and open a PR to `main`
6. Reference related issues and describe your changes clearly

### Branching & Commits

* Branch off `main`
* Use descriptive branch names and commit messages
* Keep commits atomic — one logical change per commit
* Avoid large, unrelated PRs

---

## 🧩 Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/sr-dash/SFT2D.git
   cd SFT2D
   ```
2. Create and activate a conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate sft2d
   ```
3. Install in editable mode:

   ```bash
   pip install -e .
   ```
4. Run example or test scripts to ensure everything works.

---

## ✨ Coding Standards & Style

* Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
* Use clear, descriptive names for variables, functions, and classes.
* Add docstrings for all public functions and classes (NumPy or Google style preferred).
* Keep functions small and modular.
* Avoid unnecessary memory copies; prefer in-place or view-based operations.
* Use consistent logging and informative comments.

---

## 🧪 Testing & Quality Assurance

* Add or update tests for new features and bug fixes.
* Ensure all existing tests pass before submitting a PR.
* Use the `test_data/` folder for small, reproducible test inputs.
* Validate scientific integrity (e.g., flux conservation, symmetry, correct advection behavior).
* If GitHub Actions is enabled, make sure your PR passes CI workflows.

---

## 📚 Documentation

* Update `docs/` when adding or changing functionality.
* Include usage examples in `examples/` for new features (e.g., BMR injection, polar flux calculation).
* Keep `README.md` current with installation and usage instructions.
* Write clear explanations for others to reproduce or extend your work.

---

## 🧾 Releases & Versioning

* Follow [Semantic Versioning](https://semver.org/):

  * **MAJOR** — incompatible API changes
  * **MINOR** — new features, backward-compatible
  * **PATCH** — bug fixes, backward-compatible
* Tag releases on GitHub.
* Update `CHANGELOG.md` to summarize key updates.
* Ensure tests and docs are updated before release.

---

## ⚖️ Licensing

By contributing, you agree that your contributions are licensed under the repository’s existing license (see `LICENSE`).
Do not include third-party code with incompatible licenses.

---

## 🙌 Acknowledgements

Thanks to all contributors — developers, testers, and users — for making **SFT2D** a robust and open resource for solar and stellar magnetic field research.
If you’d like formal acknowledgment (e.g., in the documentation), mention it in your PR or contact the maintainer.

---

## 📬 Contact

**Maintainer:**
Soumyaranjan Dash
📧 [sdash@nso.edu](mailto:sdash@nso.edu)

Thank you for helping improve **SFT2D** — every contribution counts!
Happy modelling! ☀️
