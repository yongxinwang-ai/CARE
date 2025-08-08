# Claude Assistant Instructions for EasyR1

## Project Overview
EasyR1 is a multi-modality RL training framework for vision-language models, forked from veRL with added vision-language support.

## Key Commands

### Setup
```bash
pip install -e .
```

### Training
```bash
# Run Qwen2.5-VL 7B GRPO training
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh

# Run Qwen2.5-VL 3B Tree GRPO training
bash examples/qwen2_5_vl_3b_geo3k_tree_grpo.sh
```

### Model Export
```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

### Testing
```bash
bash run_test.sh
```

### Code Quality
```bash
# Run linting
make style

# Run tests
make test
```

## Important Files and Directories

- `/verl/` - Core framework code
  - `/verl/trainer/` - Training logic (main.py, ray_trainer.py)
  - `/verl/models/` - Model implementations
  - `/verl/workers/` - Distributed workers
  - `/verl/utils/` - Utilities

- `/examples/` - Training scripts and configurations
  - `config.yaml` - Main configuration template
  - Various training scripts for different models

- `/scripts/` - Utility scripts
  - `model_merger.py` - Merge checkpoints to HuggingFace format

- `/checkpoints/` - Saved model checkpoints

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
