"""Utility functions for generating perturbations of sequences for PGE."""

import random
from typing import List, Optional, Union
import torch


def substitute_tokens(
    sequence: Union[List[int], torch.Tensor], 
    strength: float,
    vocab_size: int,
    special_token_ids: Optional[List[int]] = None
) -> List[int]:
    """Randomly substitutes tokens in a sequence.
    
    Args:
        sequence: Input token sequence
        strength: Probability of substituting each token (0.0 to 1.0)
        vocab_size: Size of vocabulary for random substitution
        special_token_ids: List of special token IDs to avoid
        
    Returns:
        Modified token sequence with substitutions
    """
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()
    
    if special_token_ids is None:
        special_token_ids = []
    
    result = []
    for token in sequence:
        if token in special_token_ids:
            result.append(token)
        elif random.random() < strength:
            # Substitute with a random token
            new_token = random.randint(0, vocab_size - 1)
            while new_token in special_token_ids:
                new_token = random.randint(0, vocab_size - 1)
            result.append(new_token)
        else:
            result.append(token)
    
    return result


def delete_tokens(
    sequence: Union[List[int], torch.Tensor], 
    strength: float,
    special_token_ids: Optional[List[int]] = None,
    min_length: int = 10
) -> List[int]:
    """Randomly deletes tokens from a sequence.
    
    Args:
        sequence: Input token sequence
        strength: Probability of deleting each token (0.0 to 1.0)
        special_token_ids: List of special token IDs to preserve
        min_length: Minimum length to maintain
        
    Returns:
        Modified token sequence with deletions
    """
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()
        
    if special_token_ids is None:
        special_token_ids = []
    
    result = []
    for token in sequence:
        if token in special_token_ids:
            result.append(token)
        elif random.random() >= strength or len(result) < min_length:
            result.append(token)
    
    return result


def insert_tokens(
    sequence: Union[List[int], torch.Tensor], 
    strength: float,
    neutral_tokens: Optional[List[int]] = None
) -> List[int]:
    """Randomly inserts neutral tokens into a sequence.
    
    Args:
        sequence: Input token sequence
        strength: Probability of inserting after each token (0.0 to 1.0)
        neutral_tokens: List of neutral token IDs to insert
        
    Returns:
        Modified token sequence with insertions
    """
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()
        
    if neutral_tokens is None:
        # Default neutral tokens (you may need to adjust based on tokenizer)
        neutral_tokens = []
    
    result = []
    for i, token in enumerate(sequence):
        result.append(token)
        # Don't insert at the very end
        if i < len(sequence) - 1 and neutral_tokens and random.random() < strength:
            result.append(random.choice(neutral_tokens))
    
    return result


def resample_reasoning_step(
    sequence: Union[List[int], torch.Tensor],
    model,
    tokenizer
) -> List[int]:
    """Identifies a reasoning step in Chain-of-Thought and uses model to generate alternative.
    
    Note: This is a stub function. In the actual implementation, this would:
    1. Identify CoT markers in the sequence
    2. Find a reasoning step boundary
    3. Use the model to generate an alternative continuation
    4. Replace the original step with the new one
    
    Args:
        sequence: Input token sequence
        model: Language model for generation
        tokenizer: Tokenizer
        
    Returns:
        Modified token sequence with resampled reasoning step
    """
    # For now, just return the original sequence
    # This function requires model access which we'll handle differently
    if isinstance(sequence, torch.Tensor):
        return sequence.tolist()
    return sequence