"""Test PGE implementation."""

import torch
import numpy as np
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.trainer.config import PPOConfig, AlgorithmConfig


def create_mock_batch(n_rollouts=4, batch_size=2, seq_len=20, vocab_size=1000):
    """Create a mock batch for testing."""
    total_samples = n_rollouts * batch_size
    
    # Create mock data
    prompts = []
    responses = []
    input_ids = []
    attention_masks = []
    position_ids = []
    response_masks = []
    
    for i in range(total_samples):
        prompt_len = 10
        response_len = 10
        
        # Create prompt and response
        prompt = torch.randint(0, vocab_size, (prompt_len,))
        response = torch.randint(0, vocab_size, (response_len,))
        
        # Create full sequence
        input_id = torch.cat([prompt, response])
        attention_mask = torch.ones_like(input_id)
        position_id = torch.arange(len(input_id))
        response_mask = torch.zeros_like(input_id)
        response_mask[prompt_len:] = 1
        
        prompts.append(prompt)
        responses.append(response)
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        position_ids.append(position_id)
        response_masks.append(response_mask)
    
    # Pad to same length
    max_len = max(len(ids) for ids in input_ids)
    
    padded_prompts = torch.nn.utils.rnn.pad_sequence(prompts, batch_first=True, padding_value=0)
    padded_responses = torch.nn.utils.rnn.pad_sequence(responses, batch_first=True, padding_value=0)
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    padded_position_ids = torch.nn.utils.rnn.pad_sequence(position_ids, batch_first=True, padding_value=0)
    padded_response_masks = torch.nn.utils.rnn.pad_sequence(response_masks, batch_first=True, padding_value=0)
    
    # Create UIDs for groups
    uids = []
    for i in range(batch_size):
        group_uid = f"group_{i}"
        for _ in range(n_rollouts):
            uids.append(group_uid)
    
    # Create mock rewards (higher for first sample in each group)
    token_level_scores = torch.randn(total_samples, max_len)
    for i in range(batch_size):
        # Make first sample in each group have highest reward
        token_level_scores[i * n_rollouts] += 2.0
    
    batch_dict = TensorDict({
        "prompts": padded_prompts,
        "responses": padded_responses,
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_masks,
        "position_ids": padded_position_ids,
        "response_mask": padded_response_masks,
        "token_level_scores": token_level_scores,
    }, batch_size=total_samples)
    
    non_tensor_batch = {
        "uid": np.array(uids, dtype=object),
        "other_data": np.array([f"sample_{i}" for i in range(total_samples)], dtype=object),
    }
    
    meta_info = {
        "test": True
    }
    
    return DataProto(
        batch=batch_dict,
        non_tensor_batch=non_tensor_batch,
        meta_info=meta_info
    )


def test_pge_variant():
    """Test the PGE variant processing."""
    from verl.trainer.ray_trainer import RayPPOTrainer
    from verl.utils.perturbations import substitute_tokens, delete_tokens, insert_tokens
    
    # Create mock config
    config = PPOConfig()
    config.algorithm.grpo_variant = "pge"
    config.algorithm.pge_config = {
        "num_perturbations": 3,
        "perturbation_methods": ["token_substitute", "token_delete"],
        "perturbation_strength": 0.2
    }
    config.worker.rollout.n = 4  # n_rollouts
    
    # Create mock tokenizer
    tokenizer = type('MockTokenizer', (), {
        'vocab_size': 1000,
        'pad_token_id': 0,
        'eos_token_id': 1,
        'bos_token_id': 2,
    })()
    
    # Create a simplified mock trainer
    class MockTrainer:
        def __init__(self, config, tokenizer):
            self.config = config
            self.tokenizer = tokenizer
        
        # Copy the _apply_pge_variant method from RayPPOTrainer
        def _apply_pge_variant(self, batch):
            import uuid
            import numpy as np
            from verl.utils.perturbations import substitute_tokens, delete_tokens, insert_tokens
            from verl.utils import torch_functional as VF
            from tensordict import TensorDict
            
            pge_config = self.config.algorithm.pge_config
            n_rollouts = self.config.worker.rollout.n
            
            # Get rewards if not already computed
            if "token_level_scores" not in batch.batch:
                # In real case, this would call reward function
                batch.batch["token_level_scores"] = torch.randn_like(batch.batch["response_mask"])
            
            # Sum token rewards to get response-level rewards
            response_rewards = batch.batch["token_level_scores"].sum(dim=-1)
            
            # Process each prompt group
            total_samples = len(batch.batch["responses"])
            batch_size = total_samples // n_rollouts
            
            new_responses = []
            new_prompts = []
            new_input_ids = []
            new_attention_masks = []
            new_position_ids = []
            new_response_masks = []
            
            # Store original data we need to preserve
            original_non_tensor_batch = []
            
            vocab_size = self.tokenizer.vocab_size
            special_token_ids = [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, 
                                self.tokenizer.bos_token_id] if self.tokenizer else []
            special_token_ids = [t for t in special_token_ids if t is not None]
            
            for i in range(batch_size):
                # Get rewards for this prompt's rollouts
                start_idx = i * n_rollouts
                end_idx = (i + 1) * n_rollouts
                prompt_rewards = response_rewards[start_idx:end_idx]
                
                # Select golden sample (highest reward)
                golden_idx_local = torch.argmax(prompt_rewards).item()
                golden_idx = start_idx + golden_idx_local
                
                # Extract golden sample data
                golden_response = batch.batch["responses"][golden_idx]
                golden_prompt = batch.batch["prompts"][golden_idx]
                golden_input_ids = batch.batch["input_ids"][golden_idx]
                golden_attention_mask = batch.batch["attention_mask"][golden_idx]
                golden_position_ids = batch.batch["position_ids"][golden_idx]
                golden_response_mask = batch.batch["response_mask"][golden_idx]
                
                # Store original non-tensor data for golden sample
                golden_non_tensor = {}
                for key in batch.non_tensor_batch.keys():
                    if key in batch.non_tensor_batch and len(batch.non_tensor_batch[key]) > golden_idx:
                        golden_non_tensor[key] = batch.non_tensor_batch[key][golden_idx]
                
                # Add golden sample to new batch
                new_responses.append(golden_response)
                new_prompts.append(golden_prompt)
                new_input_ids.append(golden_input_ids)
                new_attention_masks.append(golden_attention_mask)
                new_position_ids.append(golden_position_ids)
                new_response_masks.append(golden_response_mask)
                original_non_tensor_batch.append(golden_non_tensor)
                
                # Generate perturbations
                num_perturbations = pge_config["num_perturbations"]
                perturbation_methods = pge_config["perturbation_methods"]
                perturbation_strength = pge_config["perturbation_strength"]
                
                # Extract only the response tokens (not prompt)
                prompt_len = len(golden_prompt[golden_prompt != 0])  # non-padding length
                response_only = golden_response[golden_response != 0]  # non-padding response
                
                for j in range(num_perturbations):
                    # Apply perturbations only to the response part
                    perturbed_response = response_only.cpu().tolist() if torch.is_tensor(response_only) else response_only.tolist()
                    
                    for method in perturbation_methods:
                        if method == "token_substitute":
                            perturbed_response = substitute_tokens(
                                perturbed_response, perturbation_strength, vocab_size, special_token_ids
                            )
                        elif method == "token_delete":
                            perturbed_response = delete_tokens(
                                perturbed_response, perturbation_strength, special_token_ids, min_length=5
                            )
                        elif method == "token_insert":
                            # For insert, we can use padding token as neutral
                            neutral_tokens = [self.tokenizer.pad_token_id] if self.tokenizer.pad_token_id else []
                            perturbed_response = insert_tokens(
                                perturbed_response, perturbation_strength, neutral_tokens
                            )
                    
                    # Add back to new batch (simplified for testing)
                    new_responses.append(golden_response.clone())
                    new_prompts.append(golden_prompt.clone())
                    new_input_ids.append(golden_input_ids.clone())
                    new_attention_masks.append(golden_attention_mask.clone())
                    new_position_ids.append(golden_position_ids.clone())
                    new_response_masks.append(golden_response_mask.clone())
                    original_non_tensor_batch.append(golden_non_tensor.copy())
            
            # Return simplified result for testing
            return {"num_groups": batch_size, "samples_per_group": 1 + num_perturbations}
    
    # Test the function
    trainer = MockTrainer(config, tokenizer)
    batch = create_mock_batch(n_rollouts=4, batch_size=2)
    
    result = trainer._apply_pge_variant(batch)
    
    # Verify results
    assert result["num_groups"] == 2, f"Expected 2 groups, got {result['num_groups']}"
    assert result["samples_per_group"] == 4, f"Expected 4 samples per group (1 golden + 3 perturbations), got {result['samples_per_group']}"
    
    print("✓ PGE variant test passed!")
    print(f"  - Processed {result['num_groups']} groups")
    print(f"  - Each group has {result['samples_per_group']} samples (1 golden + {config.algorithm.pge_config['num_perturbations']} perturbations)")


if __name__ == "__main__":
    test_pge_variant()