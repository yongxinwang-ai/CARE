"""
TreeGRPO implementation for EasyR1.

Author: Yongxin Wang
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from ..protocol import DataProto
from . import core_algos


class SimpleTreeGRPOMixin:
    """TreeGRPO mixin for hierarchical advantage computation."""
    
    def _compute_batch_tree_advantage_simple(self, batch: DataProto) -> Tuple[DataProto, float, int]:
        """
        Compute TreeGRPO advantages with hierarchical normalization.
        
        Args:
            batch: DataProto containing generated responses and rewards
            
        Returns:
            batch: Updated batch with advantages and returns
            accuracy: Fraction of responses with positive reward
            response_tokens: Total number of response tokens
        """
        
        # Get config values
        step_length = self.config.worker.rollout.step_length
        max_response_length = self.config.data.max_response_length
        max_tree_depth = max_response_length // step_length
        
        # Extract batch info
        batch_size = batch.batch['input_ids'].shape[0]
        n_rollouts = self.config.worker.rollout.n
        
        # TreeGRPO generates multiple responses per prompt.
        # Each group of n_rollouts consecutive responses belongs to the same prompt.
        
        # Group responses by original prompt index
        responses_by_prompt = defaultdict(list)
        for idx in range(batch_size):
            prompt_idx = idx // n_rollouts
            responses_by_prompt[prompt_idx].append(idx)
        
        # Process each tree level
        all_advantages = torch.zeros_like(batch.batch['responses'], dtype=torch.float32)
        all_returns = torch.zeros_like(batch.batch['responses'], dtype=torch.float32)
        
        for depth in range(1, max_tree_depth + 1):
            start_idx = (depth - 1) * step_length
            end_idx = min(depth * step_length, max_response_length)
            
            
            # For each prompt, compute advantages across its responses
            for prompt_idx, response_indices in responses_by_prompt.items():
                # Get rewards for this depth from all sibling responses
                sibling_rewards = []
                valid_indices = []
                
                for resp_idx in response_indices:
                    # Check if response is still valid at this depth
                    response_mask = batch.batch['response_mask'][resp_idx]
                    if response_mask[start_idx:end_idx].sum() > 0:
                        # Get reward for this segment
                        if 'token_level_rewards' in batch.batch:
                            segment_reward = batch.batch['token_level_rewards'][resp_idx, start_idx:end_idx].sum().item()
                        else:
                            segment_reward = batch.batch['token_level_scores'][resp_idx, start_idx:end_idx].sum().item()
                        sibling_rewards.append(segment_reward)
                        valid_indices.append(resp_idx)
                
                # Normalize advantages across sibling responses
                if len(sibling_rewards) > 1:
                    rewards_array = np.array(sibling_rewards)
                    mean_reward = rewards_array.mean()
                    std_reward = rewards_array.std()
                    
                    # Normalize advantages
                    for i, resp_idx in enumerate(valid_indices):
                        if std_reward > 1e-8:
                            advantage = (sibling_rewards[i] - mean_reward) / (std_reward + 1e-8)
                        else:
                            advantage = 0.0
                        
                        # Apply advantage to all tokens in this segment
                        all_advantages[resp_idx, start_idx:end_idx] = advantage
                        all_returns[resp_idx, start_idx:end_idx] = sibling_rewards[i]
        
        # Store computed advantages and returns in batch
        batch.batch['advantages'] = all_advantages
        batch.batch['returns'] = all_returns
        
        # Compute accuracy (responses with positive total reward)
        total_rewards = batch.batch.get('token_level_rewards', batch.batch['token_level_scores']).sum(dim=1)
        accuracy = (total_rewards > 0).float().mean().item()
        
        # Count response tokens
        response_tokens = batch.batch['response_mask'].sum().item()
        
        return batch, accuracy, response_tokens