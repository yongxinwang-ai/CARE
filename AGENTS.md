# Repository Guidelines

## Project Structure & Module Organization
- `verl/`: Core library (trainers, workers, models, utils, protocol).
- `examples/`: Ready-to-run training scripts and configs (e.g., `qwen2_5_vl_7b_geo3k_grpo.sh`).
- `tests/`: Pytest suite for units and light integrations.
- `scripts/`: Utilities such as `model_merger.py`.
- `assets/`, `docs/`: Static assets and design notes.
- `checkpoints/`: Local outputs; do not commit artifacts or logs.

## Build, Test, and Development Commands
- `pip install -e .`: Editable install for local development.
- `make build`: Build sdist/wheel via `setup.py`.
- `make test` or `pytest -vv tests/`: Run tests.
- `make style`: Auto-fix with Ruff (lint + format).
- `make quality`: Lint/format checks (no writes).
- `make commit`: Install and run pre-commit on all files.
- Example run: `bash examples/qwen2_5_vl_7b_geo3k_grpo.sh` (see README for variants).

## Coding Style & Naming Conventions
- Python 3.9+, 4-space indent, max line length 119, double quotes preferred.
- Tools: `ruff` for linting/formatting (see `pyproject.toml`).
- Naming: modules/functions/variables `snake_case`; classes `CamelCase`.
- Example config/script naming: `<model>_<size>_<dataset>_<algo>.sh`.

## Testing Guidelines
- Framework: `pytest` (tests live in `tests/`, e.g., `tests/test_dataset.py`).
- Naming: files `test_*.py`, tests `test_*`.
- Run locally: `pytest -vv tests/` (no GPU required for unit tests; keep mocks/lightweight fixtures).
- Add unit tests for utilities and dataset transforms; add minimal integration tests for trainers/workers.

## Commit & Pull Request Guidelines
- History shows short, present-tense messages (e.g., "pge upd"). Prefer clearer scope: `trainer: fix step overflow`.
- PRs must include: summary, rationale, affected modules, sample run command, and before/after metrics or logs (if training-related). Link related issues.
- CI hygiene: run `make style`, `make quality`, and `make test` before pushing; ensure pre-commit passes.

## Security & Configuration Tips
- Do not commit checkpoints, large logs, or secrets; rely on `.gitignore`.
- Env knobs: `HF_ENDPOINT=...` mirrors; `USE_MODELSCOPE_HUB=1` to switch hubs (see `verl/__init__.py`).
- For SLURM/multi-node, keep SBATCH headers in personal scripts or `examples/`; document cluster-specific settings in PRs.

## Development Notes

- See @ROADMAP.md to check the tasks.
- The project uses Ray for distributed training
- Supports multiple vision-language models (Qwen2/2.5-VL)
- Configuration is based on OmegaConf with hierarchical settings
- Custom reward functions can be implemented in `/examples/reward_function/`
- Test the unit after the implementation

## Common Tasks

### Modifying Training Configuration
Edit `examples/config.yaml` or the specific training script

### Adding Custom Reward Functions
Create new Python files in `/examples/reward_function/`

### Debugging Training
Check logs in the output directory specified in the config