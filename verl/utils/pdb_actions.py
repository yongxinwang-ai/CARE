"""
Primal-Dual Budgeter (PDB) Action Space

Defines the simplified action space with ANS (direct answer) and TEXT-k (CoT).
According to ROADMAP: "action set当前只包含直接回答和cot两种形式"
"""

import torch
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ActionType(Enum):
    """Simplified action types for PDB."""
    ANS = "direct_answer"      # Direct answer without CoT
    TEXT_COT = "chain_of_thought"  # Chain-of-thought reasoning


class Action:
    """Represents a single action in PDB."""
    
    def __init__(self, action_type: ActionType, text_tokens: int = 0):
        """
        Initialize an action.
        
        Args:
            action_type: Type of action (ANS or TEXT_COT)
            text_tokens: Number of text tokens for this action
        """
        self.action_type = action_type
        self.text_tokens = text_tokens
        
    def get_visual_cost(self, config: Dict) -> float:
        """
        Get visual cost of this action.
        For simplified version, both ANS and TEXT_COT have no additional visual cost.
        
        Args:
            config: PDB configuration
            
        Returns:
            Visual cost (0 for text-only actions)
        """
        return 0.0
    
    def get_text_cost(self, config: Dict) -> float:
        """
        Get text cost of this action.
        
        Args:
            config: PDB configuration
            
        Returns:
            Text cost in token-equivalent units
        """
        cost_per_token = config.get("cost_per_text_token", 1.0)
        return self.text_tokens * cost_per_token
    
    def __repr__(self):
        return f"Action({self.action_type.value}, tokens={self.text_tokens})"


class ActionSpace:
    """Manages the action space for PDB."""
    
    def __init__(self, config: Dict):
        """
        Initialize action space.
        
        Args:
            config: PDB configuration
        """
        self.config = config
        
        # Define typical token counts for different action types
        # These are estimates - actual counts depend on generation
        self.ans_tokens = 50  # Short direct answer
        self.cot_tokens = 200  # Longer CoT reasoning
        
    def get_available_actions(self) -> List[Action]:
        """
        Get list of available actions.
        
        Returns:
            List of available actions
        """
        actions = []
        
        # Direct answer action
        actions.append(Action(ActionType.ANS, self.ans_tokens))
        
        # Chain-of-thought action
        actions.append(Action(ActionType.TEXT_COT, self.cot_tokens))
        
        return actions
    
    def estimate_action_gain(self, action: Action, context: Optional[Dict] = None) -> float:
        """
        Estimate information gain from an action.
        Simplified version - returns fixed estimates.
        
        Args:
            action: The action to evaluate
            context: Optional context for gain estimation
            
        Returns:
            Estimated information gain
        """
        # Simple heuristic: CoT provides more information gain than direct answer
        if action.action_type == ActionType.TEXT_COT:
            return 0.8  # Higher gain from reasoning
        elif action.action_type == ActionType.ANS:
            return 0.4  # Lower gain from direct answer
        else:
            return 0.0
            
    def select_best_action(self, pdb_controller, context: Optional[Dict] = None) -> Tuple[Optional[Action], float]:
        """
        Select best action based on gain minus cost.
        
        Args:
            pdb_controller: PDBController instance for computing action values
            context: Optional context for action selection
            
        Returns:
            (best_action, action_value) or (None, value) if should stop
        """
        actions = self.get_available_actions()
        
        best_action = None
        best_value = float('-inf')
        
        for action in actions:
            # Estimate gain and costs
            gain = self.estimate_action_gain(action, context)
            visual_cost = action.get_visual_cost(self.config)
            text_cost = action.get_text_cost(self.config)
            
            # Compute action value
            value = pdb_controller.compute_action_value(gain, visual_cost, text_cost)
            
            if value > best_value:
                best_value = value
                best_action = action
                
        # Check stopping condition
        if pdb_controller.should_stop(best_value):
            return None, best_value
            
        return best_action, best_value


def estimate_trajectory_costs(responses: torch.Tensor, tokenizer, config: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate visual and text costs for generated trajectories.
    
    Args:
        responses: Generated response token IDs (batch_size, seq_len)
        tokenizer: Tokenizer for decoding
        config: PDB configuration
        
    Returns:
        (visual_costs, text_costs): Cost tensors for each trajectory
    """
    batch_size = responses.shape[0]
    visual_costs = torch.zeros(batch_size)
    text_costs = torch.zeros(batch_size)
    
    cost_per_text_token = config.get("cost_per_text_token", 1.0)
    
    for i in range(batch_size):
        # Count actual response tokens (non-padding)
        response = responses[i]
        valid_tokens = (response != tokenizer.pad_token_id).sum().item()
        
        # For simplified version: only text cost, no visual cost
        text_costs[i] = valid_tokens * cost_per_text_token
        
        # Visual cost is 0 for text-only actions
        visual_costs[i] = 0.0
        
    return visual_costs, text_costs