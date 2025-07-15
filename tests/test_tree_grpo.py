#!/usr/bin/env python3
"""
Test script for TreeGRPO implementation in EasyR1.

This test validates:
1. TreeNode data structure
2. Tree-based advantage computation
3. Integration with the training pipeline
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

# Import TreeGRPO components
from verl.trainer.tree import TreeNode
from verl.trainer.core_algos import compute_tree_grpo_outcome_advantage, AdvantageEstimator
from verl.protocol import DataProto


def test_tree_node():
    """Test TreeNode initialization and basic operations."""
    print("Testing TreeNode...")
    
    # Create a sample batch dict
    batch_dict = {
        'input_ids': torch.ones((1, 512), dtype=torch.long),
        'attention_mask': torch.ones((1, 512), dtype=torch.long),
        'position_ids': torch.arange(512).unsqueeze(0),
        'data_source': ['test'],
        'ability': ['math'],
        'reward_model': ['test_rm'],
        'extra_info': [''],
        'index': [0]
    }
    
    # Create root node
    root = TreeNode(
        batch_dict=batch_dict,
        depth=0,
        prompt_length=512,
        max_response_length=1280,
        step_length=256,
        eos_token_id=151643
    )
    
    assert root.depth == 0
    assert not root.is_end
    assert len(root.children) == 0
    assert root.prompt_length == 512
    assert root.max_response_length == 1280
    assert root.step_length == 256
    
    # Test adding children
    child_batch = batch_dict.copy()
    child_batch['input_ids'] = torch.ones((1, 768), dtype=torch.long)  # 512 + 256
    child_batch['attention_mask'] = torch.ones((1, 768), dtype=torch.long)
    child_batch['position_ids'] = torch.arange(768).unsqueeze(0)
    child_batch['responses'] = torch.ones((1, 256), dtype=torch.long)
    
    child = TreeNode(
        batch_dict=child_batch,
        depth=1,
        prompt_length=512,
        max_response_length=1280,
        step_length=256,
        reward=1.5,
        eos_token_id=151643
    )
    
    root.add_child(child)
    assert len(root.children) == 1
    assert root.children[0].depth == 1
    assert root.children[0].E_reward == 1.5
    
    print("✓ TreeNode test passed")


def test_tree_grpo_advantage():
    """Test TreeGRPO advantage computation."""
    print("\nTesting TreeGRPO advantage computation...")
    
    # Create sample data
    batch_size = 1
    response_length = 256
    
    token_level_rewards = torch.zeros((batch_size, response_length))
    token_level_rewards[0, -1] = 2.0  # Reward at last token
    
    response_mask = torch.ones((batch_size, response_length))
    index = torch.tensor([0])
    
    # Test with multiple brother rewards
    # Note: brother_rewards should include the current node's reward
    current_reward = 2.0
    brother_rewards = [1.0, 2.0, 3.0, 0.5]  # These are the rewards from all siblings
    
    advantages, returns = compute_tree_grpo_outcome_advantage(
        token_level_rewards,
        response_mask,
        index,
        brother_rewards=brother_rewards
    )
    
    # Check output shapes
    assert advantages.shape == (batch_size, response_length)
    assert returns.shape == (batch_size, response_length)
    
    # Check normalization
    mean_reward = np.mean(brother_rewards)
    std_reward = np.std(brother_rewards)
    torch_mean = torch.tensor(brother_rewards).mean()
    torch_std = torch.tensor(brother_rewards).std()
    expected_adv = (current_reward - mean_reward) / (std_reward + 1e-6)
    expected_adv_torch = (current_reward - torch_mean) / (torch_std + 1e-6)
    
    print(f"Current node reward: {current_reward}")
    print(f"Brother rewards: {brother_rewards}")
    print(f"Mean reward (numpy): {mean_reward}, (torch): {torch_mean}")
    print(f"Std reward (numpy): {std_reward}, (torch): {torch_std}")
    print(f"Expected advantage (numpy): {expected_adv}")
    print(f"Expected advantage (torch): {expected_adv_torch}")
    print(f"Actual advantage: {advantages[0, 0].item()}")
    
    # The advantage should be constant across all tokens for TreeGRPO
    # Use torch's expected value since our implementation uses torch.std()
    assert abs(advantages[0, 0].item() - expected_adv_torch.item()) < 1e-5
    
    # Test with single node (no normalization possible)
    brother_rewards_single = [2.0]  # Only current node
    advantages_single, returns_single = compute_tree_grpo_outcome_advantage(
        token_level_rewards,
        response_mask,
        index,
        brother_rewards=brother_rewards_single
    )
    
    assert advantages_single.shape == (batch_size, response_length)
    # With only one reward, it should fall back to GRPO but we can't test that here
    # because GRPO requires multiple samples from the same prompt
    
    print("✓ TreeGRPO advantage computation test passed")


def test_tree_batch_formatting():
    """Test batch formatting for tree nodes."""
    print("\nTesting tree batch formatting...")
    
    # Create a node with data
    batch_dict = {
        'input_ids': torch.ones((1, 768), dtype=torch.long),
        'attention_mask': torch.ones((1, 768), dtype=torch.long),
        'position_ids': torch.arange(768).unsqueeze(0),
        'prompts': torch.ones((1, 512), dtype=torch.long),
        'responses': torch.ones((1, 256), dtype=torch.long),
        'token_level_rewards': torch.zeros((1, 256)),
        'token_level_scores': torch.zeros((1, 256)),
        'advantages': torch.zeros((1, 256)),
        'returns': torch.zeros((1, 256)),
        'data_source': ['test'],
        'ability': ['math'],
        'reward_model': ['test_rm'],
        'extra_info': [''],
        'index': [0]
    }
    
    node = TreeNode(
        batch_dict=batch_dict,
        depth=1,
        prompt_length=512,
        max_response_length=1280,
        step_length=256
    )
    
    # Test input batch formatting
    input_batch = node.format_node_input_batch()
    assert isinstance(input_batch, DataProto)
    assert 'input_ids' in input_batch.batch
    assert input_batch.batch['input_ids'].shape == (1, 768)
    
    # Test train batch formatting
    pad_token_id = 151643
    train_batch = node.format_node_train_batch(pad_token_id)
    assert isinstance(train_batch, DataProto)
    
    # Check padding
    max_length = 512 + (1280 // 256) * 256  # prompt_length + max_depth * step_length
    assert train_batch.batch['input_ids'].shape == (1, max_length)
    assert train_batch.batch['attention_mask'].shape == (1, max_length)
    
    print("✓ Tree batch formatting test passed")


def test_advantage_estimator_enum():
    """Test that TreeGRPO is properly added to AdvantageEstimator enum."""
    print("\nTesting AdvantageEstimator enum...")
    
    assert hasattr(AdvantageEstimator, 'TREE_GRPO')
    assert AdvantageEstimator.TREE_GRPO == "tree_grpo"
    
    # Check all estimators
    all_estimators = list(AdvantageEstimator)
    assert AdvantageEstimator.TREE_GRPO in all_estimators
    
    print("✓ AdvantageEstimator enum test passed")


def run_all_tests():
    """Run all TreeGRPO tests."""
    print("=" * 50)
    print("Running TreeGRPO Tests")
    print("=" * 50)
    
    test_tree_node()
    test_tree_grpo_advantage()
    test_tree_batch_formatting()
    test_advantage_estimator_enum()
    
    print("\n" + "=" * 50)
    print("All TreeGRPO tests passed! ✅")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()