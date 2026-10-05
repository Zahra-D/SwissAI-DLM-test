# Contributing to UNI-D²

Thank you for your interest in contributing! We aim to build a standard OSS library for discrete diffusion. Please keep code minimal, clean, and concise.

## Development Setup

Clone the repository and install the package in editable mode:

```bash
pip install -e .
```

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. If you add a new feature, you must add corresponding documentation.
3. Submit a Pull Request with a clear description of your changes.

## Documentation

We use [MkDocs](https://www.mkdocs.org/) with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) for documentation.

To build the documentation locally:

1. Install documentation dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Serve the documentation:
   ```bash
   mkdocs serve
   ```

   This will start a local server at `http://127.0.0.1:8000/` where you can preview changes.

3. To build the static site:
   ```bash
   mkdocs build
   ```
