# AGENTS.md

This file provides guidance to the AI agent when working with code in this repository.

## Communication

- 与用户交流、写文档和注释时使用中文。
- Git commit messages and PR text: English, with prefixes `feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:`.

## Environment

- Python >= 3.12; `uv` is the source of truth. Install deps with `uv sync`.
- `torch`/`torchvision` are pinned to a custom PyTorch CUDA index (`pytorch-cu130`) via `[tool.uv.sources]` — do not `pip install` them separately.
- Model downloads default to the hf-mirror site.

## Commands

- Run the CLI via `uv run main.py <command>` or the installed `psorad <command>`.
- Subcommands: `preprocess`, `download-models`, `train`, `attack` (see README for flags).
- Batch attack tool: `uv run python tools/batch_parallel_attack.py ...`.

## Quality gate (run before marking work done)

```bash
ruff check .
mypy src
pytest
```

- `ruff format` handles formatting; ruff `line-length` is **240** (not the default 88).
- pre-commit config exists; enable with `pre-commit install` (optional).

## Layout & conventions

- Package lives under `src/psorad/`; `main.py` is a thin wrapper over `psorad.cli:main`.
- Respect module boundaries: keep algorithm code (`attack/`, `models/`) decoupled from CLI details.
- Note: training code now lives in `src/psorad/models/train.py` (the old `trainers/` module was removed).
- Branch naming: `feat/` `fix/` `docs/` `refactor/` `<short-description>`.

## Do not commit

- Real dataset samples under `dataset/`.
- Large model weights under `model/pretrained_model/` or `model/trained_classifier/`.
