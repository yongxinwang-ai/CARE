"""
TreeGRPO Trainer extension for RayPPOTrainer.

This module provides TreeGRPO-specific methods that extend the base RayPPOTrainer
to support tree-based generation and training.
"""

import torch
import numpy as np
import ray
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..utils import torch_functional as VF
from . import core_algos
from .tree import TreeNode


class TreeGRPOMixin:
    """Mixin class that adds TreeGRPO functionality to RayPPOTrainer."""
    
    def _init_tree_params(self):
        """Initialize TreeGRPO-specific parameters."""
        self.max_tree_depth = self.config.data.max_response_length // self.config.worker.rollout.step_length
        if self.config.data.max_response_length % self.config.worker.rollout.step_length != 0:
            raise ValueError(
                f"max_response_length ({self.config.data.max_response_length}) must be divisible by "
                f"step_length ({self.config.worker.rollout.step_length})"
            )
        print(f"TreeGRPO: Max tree depth = {self.max_tree_depth}, Step length = {self.config.worker.rollout.step_length}")
    
    def _compute_batch_tree_advantage(self, input_batch: DataProto) -> Tuple[DataProto, float, int]:
        """
        Compute advantages for a batch using tree-based generation.
        
        Args:
            input_batch: Input batch containing prompts
            
        Returns:
            train_batch: Batch ready for training with computed advantages
            accuracy: Training accuracy
            total_response_tokens: Total number of response tokens generated
        """
        import sys
        import os
        print("TreeGRPO: _compute_batch_tree_advantage called", flush=True)
        print(f"TreeGRPO: Process PID: {os.getpid()}", flush=True)
        sys.stdout.flush()
        
        batch_size = input_batch.batch['input_ids'].shape[0]
        
        # Initialize tree structure - one tree per prompt
        level_node_list = [[] for _ in range(self.max_tree_depth + 1)]
        
        print(f"TreeGRPO: Starting tree generation with batch_size={batch_size}, max_depth={self.max_tree_depth}")
        
        # Create root nodes
        for idx in range(batch_size):
            # Build batch dict with tensor data
            batch_dict = {
                'input_ids': input_batch.batch['input_ids'][idx:idx+1, :],
                'attention_mask': input_batch.batch['attention_mask'][idx:idx+1, :],
            }
            
            # Add position_ids if present
            if 'position_ids' in input_batch.batch:
                batch_dict['position_ids'] = input_batch.batch['position_ids'][idx:idx+1, :]
            
            # Add non-tensor data only if present
            if input_batch.non_tensor_batch:
                for key in ['data_source', 'ability', 'reward_model', 'extra_info', 'raw_prompt_ids']:
                    if key in input_batch.non_tensor_batch and len(input_batch.non_tensor_batch[key]) > idx:
                        batch_dict[key] = input_batch.non_tensor_batch[key][idx:idx+1]
            
            # Always add index
            batch_dict['index'] = [idx]
            
            root_node = TreeNode(
                batch_dict=batch_dict,
                depth=0,
                prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                step_length=self.config.worker.rollout.step_length,
                eos_token_id=self.tokenizer.eos_token_id
            )
            level_node_list[0].append(root_node)
        
        # Generate tree level by level
        for depth in range(1, self.max_tree_depth + 1):
            print(f"TreeGRPO: Generating depth {depth}/{self.max_tree_depth}")
            print(f"TreeGRPO: Current nodes at depth {depth-1}: {len(level_node_list[depth-1])}")
            
            # Format batch for current depth
            depth_batch, pad_size = self._format_depth_input_batch(level_node_list[depth - 1])
            if depth_batch is None:
                print(f"TreeGRPO: No valid nodes at depth {depth-1}, stopping tree generation", flush=True)
                break  # All nodes have ended
            
            # Generate next depth
            try:
                print(f"TreeGRPO: Generating sequences for {depth_batch.batch['input_ids'].shape[0]} nodes")
                # For TreeGRPO, we need to temporarily set the response_length to step_length
                # Save original response_length
                original_response_length = self.config.data.max_response_length
                
                # HACK: Temporarily modify config for step-wise generation
                # This is needed because vLLM rollout uses config.response_length
                print(f"TreeGRPO: Using step_length={self.config.worker.rollout.step_length} for generation")
                
                # Add step length info to meta_info
                depth_batch.meta_info = depth_batch.meta_info or {}
                depth_batch.meta_info['max_tokens'] = self.config.worker.rollout.step_length
                
                print(f"TreeGRPO: Calling generate_sequences...", flush=True)
                gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(depth_batch)
                print(f"TreeGRPO: generate_sequences returned", flush=True)
                
                gen_batch_output = unpad_dataproto(gen_batch_output, pad_size)
                print(f"TreeGRPO: Generated {gen_batch_output.batch['input_ids'].shape[0]} sequences", flush=True)
                
                # Debug: Check generated output
                if 'responses' in gen_batch_output.batch:
                    print(f"TreeGRPO: Response shape: {gen_batch_output.batch['responses'].shape}", flush=True)
                
                # Compute rewards for generated sequences
                print(f"TreeGRPO: Computing rewards...", flush=True)
                reward_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(gen_batch_output))
                print(f"TreeGRPO: Computed rewards shape: {reward_tensor.shape}", flush=True)
            except Exception as e:
                print(f"TreeGRPO: Error at depth {depth}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # Add generated nodes as children
            self._add_children_nodes(level_node_list, depth, gen_batch_output, reward_tensor)
            print(f"TreeGRPO: Added {len(level_node_list[depth])} nodes at depth {depth}")
        
        # Print tree statistics
        total_nodes = sum(len(level) for level in level_node_list)
        print(f"TreeGRPO: Tree generation complete. Total nodes: {total_nodes}")
        for i, level in enumerate(level_node_list):
            if level:
                print(f"  Depth {i}: {len(level)} nodes")
        
        # Compute expected rewards for all nodes
        print("TreeGRPO: Computing expected rewards...")
        for root in level_node_list[0]:
            if not root.is_end:
                self._traversal_reward(root)
        
        # Compute advantages for all nodes
        print("TreeGRPO: Computing advantages...")
        for root in level_node_list[0]:
            if not root.is_end:
                self._traversal_advantage(root, [])
        
        # Format training batch
        print(f"TreeGRPO: Formatting training batch...")
        train_batch = self._format_train_batch(level_node_list)
        
        if train_batch is None:
            print("TreeGRPO: WARNING - No training batch generated (all nodes pruned)")
            # Return empty batch with zero metrics
            return None, 0.0, 0
        
        print(f"TreeGRPO: Training batch size: {train_batch.batch['input_ids'].shape[0]}")
        
        # Compute accuracy and token count
        accuracy = self._compute_tree_accuracy(level_node_list)
        total_tokens = self._count_response_tokens(level_node_list)
        
        print(f"TreeGRPO: Accuracy: {accuracy:.2f}, Total tokens: {total_tokens}")
        
        return train_batch, accuracy, total_tokens
    
    def _format_depth_input_batch(self, nodes: List[TreeNode]) -> Tuple[Optional[DataProto], int]:
        """Format input batch for nodes at a specific depth."""
        valid_nodes = [node for node in nodes if not node.is_end]
        ended_nodes = [node for node in nodes if node.is_end]
        print(f"TreeGRPO: Valid nodes: {len(valid_nodes)}, Ended nodes: {len(ended_nodes)}", flush=True)
        
        if not valid_nodes:
            return None, 0
        
        # Duplicate each node n times for multiple rollouts
        batch_list = []
        for node in valid_nodes:
            for _ in range(self.config.worker.rollout.n):
                batch_list.append(node.format_node_input_batch())
        
        depth_batch = DataProto.concat(batch_list)
        depth_batch, pad_size = pad_dataproto_to_divisor(depth_batch, self.actor_rollout_ref_wg.world_size)
        
        return depth_batch, pad_size
    
    def _add_children_nodes(self, level_node_list: List[List[TreeNode]], depth: int, 
                           gen_batch: DataProto, rewards: torch.Tensor):
        """Add generated sequences as child nodes to their parent nodes."""
        parent_nodes = [node for node in level_node_list[depth - 1] if not node.is_end]
        
        idx = 0
        for parent_idx, parent in enumerate(parent_nodes):
            parent.child_rewards = []
            
            for n in range(self.config.worker.rollout.n):
                child_reward = rewards[idx].sum().item()
                parent.child_rewards.append(child_reward)
                
                # Create child node
                start_idx = self.config.data.max_prompt_length + self.config.worker.rollout.step_length * (depth - 1)
                end_idx = start_idx + self.config.worker.rollout.step_length
                
                # Build child batch dict
                child_batch_dict = {
                    'input_ids': gen_batch.batch['input_ids'][idx:idx+1, :end_idx],
                    'attention_mask': gen_batch.batch['attention_mask'][idx:idx+1, :end_idx],
                    'prompts': gen_batch.batch['input_ids'][idx:idx+1, :start_idx],
                    'responses': gen_batch.batch['responses'][idx:idx+1],
                }
                
                # Add position_ids if present
                if 'position_ids' in gen_batch.batch:
                    child_batch_dict['position_ids'] = gen_batch.batch['position_ids'][idx:idx+1, :end_idx]
                
                # Add non-tensor data only if present
                if gen_batch.non_tensor_batch:
                    for key in ['data_source', 'ability', 'reward_model', 'extra_info', 'raw_prompt_ids']:
                        if key in gen_batch.non_tensor_batch and len(gen_batch.non_tensor_batch[key]) > idx:
                            child_batch_dict[key] = gen_batch.non_tensor_batch[key][idx:idx+1]
                
                # Always add index
                child_batch_dict['index'] = [parent_idx]
                
                child_node = TreeNode(
                    batch_dict=child_batch_dict,
                    depth=depth,
                    prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    step_length=self.config.worker.rollout.step_length,
                    reward=child_reward,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                parent.add_child(child_node)
                level_node_list[depth].append(child_node)
                idx += 1
    
    def _traversal_reward(self, node: TreeNode) -> float:
        """Recursively compute expected rewards for tree nodes."""
        if node.is_end or node.depth == self.max_tree_depth:
            return node.E_reward
        
        assert len(node.children) == self.config.worker.rollout.n
        
        # Compute expected reward as mean of children rewards
        reward = 0
        for child in node.children:
            child_reward = self._traversal_reward(child)
            reward += child_reward
        
        node.E_reward = reward / len(node.children)
        return node.E_reward
    
    def _traversal_advantage(self, node: TreeNode, brother_rewards: List[float]):
        """Recursively compute advantages for tree nodes."""
        node.bro_rewards = brother_rewards
        
        if node.depth != 0:
            # Compute token-level rewards and scores
            response_length = self.config.worker.rollout.step_length
            node.batch_dict['token_level_rewards'] = torch.zeros((1, response_length))
            node.batch_dict['token_level_rewards'][0, -1] = node.E_reward
            node.batch_dict['token_level_scores'] = node.batch_dict['token_level_rewards'].clone()
            
            # Compute advantage using TreeGRPO
            response_mask = node.batch_dict['attention_mask'][:, -response_length:]
            advantages, returns = core_algos.compute_tree_grpo_outcome_advantage(
                node.batch_dict['token_level_rewards'],
                response_mask,
                torch.tensor([0]),  # Single index for this node
                brother_rewards=node.bro_rewards
            )
            
            node.batch_dict['advantages'] = advantages
            node.batch_dict['returns'] = returns
        
        if node.is_end or node.depth == self.max_tree_depth:
            return
        
        # Recursively compute advantages for children
        for child in node.children:
            self._traversal_advantage(child, node.child_rewards)
    
    def _format_train_batch(self, level_node_list: List[List[TreeNode]]) -> Optional[DataProto]:
        """Format all tree nodes into a training batch."""
        train_batch_list = []
        
        # Collect all non-root nodes for training
        for depth in range(1, self.max_tree_depth + 1):
            if len(level_node_list[depth]) == 0:
                break
            
            for node in level_node_list[depth]:
                # Prune nodes with low reward variance among siblings
                if max(node.bro_rewards) - min(node.bro_rewards) < 0.1:
                    continue
                
                train_batch_list.append(node.format_node_train_batch(self.tokenizer.pad_token_id))
        
        if not train_batch_list:
            return None
        
        return DataProto.concat(train_batch_list)
    
    def _compute_tree_accuracy(self, level_node_list: List[List[TreeNode]]) -> float:
        """Compute accuracy across all tree nodes."""
        correct = 0
        total = 0
        
        for depth in range(1, self.max_tree_depth + 1):
            for node in level_node_list[depth]:
                if 'token_level_rewards' in node.batch_dict:
                    # Check if reward is positive (correct)
                    if node.E_reward > 0:
                        correct += 1
                    total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _count_response_tokens(self, level_node_list: List[List[TreeNode]]) -> int:
        """Count total response tokens across all tree nodes."""
        total_tokens = 0
        
        for depth in range(1, self.max_tree_depth + 1):
            for node in level_node_list[depth]:
                if 'responses' in node.batch_dict:
                    total_tokens += node.batch_dict['attention_mask'][:, -self.config.worker.rollout.step_length:].sum().item()
        
        return total_tokens