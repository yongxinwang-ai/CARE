#!/usr/bin/env python3
"""Simple test to verify PGE and CGSG implementations."""

from verl.utils.perturbations import substitute_tokens, delete_tokens, insert_tokens

def test_perturbations():
    """Test perturbation functions."""
    print("Testing perturbation functions...")
    
    # Test sequence
    sequence = [100, 200, 300, 400, 500]
    vocab_size = 1000
    
    # Test substitute_tokens
    perturbed = substitute_tokens(sequence, strength=0.5, vocab_size=vocab_size)
    print(f"Original: {sequence}")
    print(f"Substituted: {perturbed}")
    assert len(perturbed) == len(sequence)
    
    # Test delete_tokens
    deleted = delete_tokens(sequence, strength=0.3)
    print(f"Deleted: {deleted}")
    assert len(deleted) <= len(sequence)
    
    # Test insert_tokens (with empty neutral tokens for now)
    inserted = insert_tokens(sequence, strength=0.3, neutral_tokens=[999])
    print(f"Inserted: {inserted}")
    assert len(inserted) >= len(sequence)
    
    print("✓ Perturbation tests passed!\n")

def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    
    from verl.trainer.config import AlgorithmConfig
    from omegaconf import OmegaConf
    
    # Create config
    cfg_dict = {
        "grpo_variant": "pge",
        "pge_config": {
            "num_perturbations": 4,
            "perturbation_methods": ["token_substitute"],
            "perturbation_strength": 0.1
        },
        "cgsg_config": {
            "num_negatives": 3,
            "negative_selection_strategy": "lowest_reward",
            "loss_type": "normalized_advantage"
        }
    }
    
    cfg = OmegaConf.create(cfg_dict)
    algo_cfg = AlgorithmConfig(**cfg)
    
    print(f"GRPO variant: {algo_cfg.grpo_variant}")
    print(f"PGE config: {algo_cfg.pge_config}")
    print(f"CGSG config: {algo_cfg.cgsg_config}")
    print("✓ Configuration tests passed!\n")

if __name__ == "__main__":
    test_perturbations()
    test_config()
    print("All tests passed! 🎉")