"""
Primal-Dual Budgeter (PDB) Dual Variable Manager

Manages Lagrangian dual variables (λ_v, λ_t) for visual and text budget constraints.
Implements adaptive dual updates based on budget violations.
"""

import torch
from typing import Tuple, Optional, Dict


class DualVariableManager:
    """Manages dual variables for PDB optimization."""
    
    def __init__(self, config: Dict):
        """
        Initialize dual variable manager.
        
        Args:
            config: PDB configuration dictionary containing:
                - visual_budget: Visual token budget
                - text_budget: Text token budget  
                - lambda_v_init: Initial visual dual variable
                - lambda_t_init: Initial text dual variable
                - lambda_v_lr: Learning rate for visual dual
                - lambda_t_lr: Learning rate for text dual
        """
        self.visual_budget = config.get("visual_budget", 512)
        self.text_budget = config.get("text_budget", 160)
        
        # Initialize dual variables
        self.lambda_v = config.get("lambda_v_init", 0.02)
        self.lambda_t = config.get("lambda_t_init", 0.01)
        
        # Learning rates for dual updates
        self.lambda_v_lr = config.get("lambda_v_lr", 0.001)
        self.lambda_t_lr = config.get("lambda_t_lr", 0.001)
        
        # History tracking
        self.visual_cost_history = []
        self.text_cost_history = []
        
    def update(self, visual_cost: float, text_cost: float) -> None:
        """
        Update dual variables based on constraint violations.
        
        Uses projected gradient ascent:
        λ_v ← [λ_v + η_v(C_v - B_v)]_+
        λ_t ← [λ_t + η_t(C_t - B_t)]_+
        
        Args:
            visual_cost: Actual visual cost incurred
            text_cost: Actual text cost incurred
        """
        # Track costs
        self.visual_cost_history.append(visual_cost)
        self.text_cost_history.append(text_cost)
        
        # Compute violations
        visual_violation = visual_cost - self.visual_budget
        text_violation = text_cost - self.text_budget
        
        # Update dual variables with projection to non-negative
        self.lambda_v = max(0.0, self.lambda_v + self.lambda_v_lr * visual_violation)
        self.lambda_t = max(0.0, self.lambda_t + self.lambda_t_lr * text_violation)
        
    def get_multipliers(self) -> Tuple[float, float]:
        """
        Get current dual multipliers.
        
        Returns:
            (lambda_v, lambda_t): Current visual and text dual multipliers
        """
        return self.lambda_v, self.lambda_t
    
    def apply_sla_mode(self, sla_mode: str, sla_presets: Dict) -> None:
        """
        Apply SLA (Service Level Agreement) preset.
        
        Args:
            sla_mode: One of "strict", "normal", "relaxed"
            sla_presets: Dictionary of SLA configurations
        """
        if sla_mode not in sla_presets:
            print(f"Warning: Unknown SLA mode {sla_mode}, using current settings")
            return
            
        preset = sla_presets[sla_mode]
        self.visual_budget = preset.get("visual_budget", self.visual_budget)
        self.text_budget = preset.get("text_budget", self.text_budget)
        self.lambda_v = preset.get("lambda_v_init", self.lambda_v)
        self.lambda_t = preset.get("lambda_t_init", self.lambda_t)
        
    def get_stats(self) -> Dict:
        """
        Get statistics about dual variables and costs.
        
        Returns:
            Dictionary with current state and historical statistics
        """
        stats = {
            "lambda_v": self.lambda_v,
            "lambda_t": self.lambda_t,
            "visual_budget": self.visual_budget,
            "text_budget": self.text_budget,
        }
        
        if self.visual_cost_history:
            stats["avg_visual_cost"] = sum(self.visual_cost_history) / len(self.visual_cost_history)
            stats["visual_budget_utilization"] = stats["avg_visual_cost"] / self.visual_budget
            
        if self.text_cost_history:
            stats["avg_text_cost"] = sum(self.text_cost_history) / len(self.text_cost_history)
            stats["text_budget_utilization"] = stats["avg_text_cost"] / self.text_budget
            
        return stats


class PDBController:
    """Main controller for Primal-Dual Budgeter."""
    
    def __init__(self, config: Dict):
        """
        Initialize PDB controller.
        
        Args:
            config: PDB configuration dictionary
        """
        self.config = config
        self.dual_manager = DualVariableManager(config)
        
        # Apply SLA mode if specified
        sla_mode = config.get("sla_mode", "normal")
        sla_presets = config.get("sla_presets", {})
        if sla_mode and sla_presets:
            self.dual_manager.apply_sla_mode(sla_mode, sla_presets)
            
        # Stopping threshold for optimal stopping
        self.stopping_threshold = config.get("stopping_threshold", 0.0)
        
    def should_stop(self, max_gain_minus_cost: float) -> bool:
        """
        Determine if optimal stopping condition is met.
        
        Args:
            max_gain_minus_cost: Maximum value of Δ(a) - λ_v*c_v(a) - λ_t*c_t(a)
            
        Returns:
            True if should stop (no beneficial action), False otherwise
        """
        return max_gain_minus_cost <= self.stopping_threshold
        
    def compute_action_value(self, gain: float, visual_cost: float, text_cost: float) -> float:
        """
        Compute action value for decision making.
        
        Value = Δ(a) - λ_v*c_v(a) - λ_t*c_t(a)
        
        Args:
            gain: Information gain from action
            visual_cost: Visual cost of action
            text_cost: Text cost of action
            
        Returns:
            Action value (higher is better)
        """
        lambda_v, lambda_t = self.dual_manager.get_multipliers()
        return gain - lambda_v * visual_cost - lambda_t * text_cost