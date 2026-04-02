# qRG
A torch based toolkit for renormalization group computations.

## Developer Setup

This repository uses `uv` for environment management, `tox` for checks, and `pre-commit` for local hooks.

### Requirements

- Python 3.11
- `uv`

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Quick Start

Clone the repo and choose one PyTorch backend extra:

- `cpu`
- `cu126`
- `cu128`
- `cu129`
- `cu130`

Standard development setup with CPU PyTorch:

```bash
git clone <your-repo-url>
cd qRG
uv sync --group dev --extra cpu
uv run pre-commit install
```

If you want a CUDA backend instead, replace `cpu` with one of the CUDA extras, for example:

```bash
uv sync --group dev --extra cu128
```

Only one backend extra can be installed at a time.

### Common Commands

Run all checks:

```bash
uv run tox
```

Run pre-commit on all files:

```bash
uv run pre-commit run --all-files
```

Build the package:

```bash
uv run python -m build
```

### Minimal Install

If you only want the base package without dev tools or PyTorch extras:

```bash
uv sync
```

This installs the package and its base dependency `qten`, but not `dev` or any torch backend.

### Reinstall

Clean reinstall:

```bash
rm -rf .venv uv.lock
uv sync --group dev --extra cpu
```

Force reinstall in the existing environment:

```bash
uv sync --reinstall --group dev --extra cpu
```

### Notes

- `uv.lock` is ignored in this repository.
- Common Torch checkpoint files are ignored in git.
