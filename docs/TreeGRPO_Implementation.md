# TreeGRPO Implementation in EasyR1

## Overview

TreeGRPO (Tree-based Group Relative Policy Optimization) has been successfully integrated into the EasyR1 framework. This implementation extends the standard GRPO algorithm by generating responses in a tree structure, allowing for better exploration and credit assignment during reinforcement learning.

## Key Features

1. **Tree-based Generation**: Responses are generated incrementally in steps (default: 256 tokens) rather than all at once
2. **Hierarchical Advantage Computation**: Advantages are computed over the tree structure, considering sibling nodes at each depth
3. **Better Credit Assignment**: Rewards can be attributed to specific parts of the generation
4. **Efficient Exploration**: Multiple branches are explored at each tree depth

## Implementation Details

### Core Components

1. **`verl/trainer/tree.py`**: TreeNode data structure for managing the generation tree
2. **`verl/trainer/core_algos.py`**: Added `compute_tree_grpo_outcome_advantage()` function
3. **`verl/trainer/tree_trainer.py`**: TreeGRPOMixin providing tree-specific training methods
4. **`verl/trainer/ray_trainer.py`**: Modified to support TreeGRPO training flow
5. **`verl/workers/rollout/config.py`**: Added `step_length` parameter

### Algorithm Flow

1. Initialize tree with root nodes (one per prompt)
2. For each depth level:
   - Generate `n` children for each non-terminal node
   - Compute rewards for generated sequences
   - Add children to tree structure
3. Compute expected rewards recursively from leaves to root
4. Compute advantages using sibling node rewards for normalization
5. Format tree nodes into training batch
6. Train model on tree-generated data

### Configuration Parameters

- `algorithm.adv_estimator`: Set to `"tree_grpo"` to enable TreeGRPO
- `worker.rollout.step_length`: Length of each generation step (default: 256)
- `worker.rollout.n`: Number of children per node (default: 8)
- `data.max_response_length`: Maximum total response length

## Usage

### Running TreeGRPO Training

```bash
# Example training script
./examples/qwen2_5_vl_3b_geo3k_tree_grpo.sh
```

### Configuration Example

```yaml
algorithm:
  adv_estimator: tree_grpo
  
worker:
  rollout:
    step_length: 256
    n: 8
```

### Testing

Run the test suite to validate TreeGRPO implementation:

```bash
cd /mnt/weka/home/yongxin.wang/workspace/EasyR1
python tests/test_tree_grpo.py
```

## Advantages over Standard GRPO

1. **Incremental Generation**: Allows for early stopping and more efficient exploration
2. **Tree Structure**: Captures dependencies between partial responses
3. **Better Variance Reduction**: Normalization uses sibling nodes at the same depth
4. **Flexible Credit Assignment**: Rewards can be computed at each tree node

## Implementation Notes

- TreeGRPO requires `rollout.n > 1` to generate multiple branches
- The maximum tree depth is `max_response_length / step_length`
- Nodes with low reward variance among siblings are pruned during training
- The implementation falls back to zero advantage for single-node cases

## Future Improvements

1. Dynamic step length based on generation complexity
2. Adaptive tree pruning strategies
3. Parallel tree generation for improved efficiency
4. Support for different tree exploration strategies