# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PDB (Primal-Dual Budgeter) dual variable management and optimal stopping
"""

from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from collections import deque

from .pdb_actions import Action, ActionType, StopAction
from .pdb_costs import CostModel, BudgetTracker
from .pdb_gains import CombinedGainEstimator


class DualVariableManager:
    """Manages dual variables (Lagrangian multipliers) for budget constraints"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Apply SLA preset if specified
        sla_mode = config.get("sla_mode", "normal")
        if sla_mode in config.get("sla_presets", {}):
            preset = config["sla_presets"][sla_mode]
            config.update(preset)
        
        # Initialize dual variables
        self.lambda_v = config.get("lambda_v_init", 0.02)
        self.lambda_t = config.get("lambda_t_init", 0.01)
        
        # Learning rates
        self.lambda_v_lr = config.get("lambda_v_lr", 0.001)
        self.lambda_t_lr = config.get("lambda_t_lr", 0.001)
        
        # Budgets
        self.visual_budget = config.get("visual_budget", 512)
        self.text_budget = config.get("text_budget", 160)
        
        # History for adaptive updates
        self.history_window = 100
        self.visual_usage_history = deque(maxlen=self.history_window)
        self.text_usage_history = deque(maxlen=self.history_window)
        
        # Statistics
        self.total_episodes = 0
        self.budget_violations = {"visual": 0, "text": 0}
    
    def update(self, visual_cost: float, text_cost: float):
        """
        Update dual variables based on budget constraint violations
        Uses projected gradient ascent on the dual problem
        """
        self.total_episodes += 1
        
        # Record usage
        self.visual_usage_history.append(visual_cost)
        self.text_usage_history.append(text_cost)
        
        # Check violations
        visual_violation = visual_cost - self.visual_budget
        text_violation = text_cost - self.text_budget
        
        if visual_violation > 0:
            self.budget_violations["visual"] += 1
        if text_violation > 0:
            self.budget_violations["text"] += 1
        
        # Dual variable updates (gradient ascent on Lagrangian)
        # λ_v ← [λ_v + η_v(C_v - B_v)]⁺
        self.lambda_v = max(0, self.lambda_v + self.lambda_v_lr * visual_violation)
        # λ_t ← [λ_t + η_t(C_t - B_t)]⁺
        self.lambda_t = max(0, self.lambda_t + self.lambda_t_lr * text_violation)
        
        # Adaptive learning rate adjustment
        if self.total_episodes % 50 == 0 and self.total_episodes > 0:
            self._adjust_learning_rates()
    
    def _adjust_learning_rates(self):
        """Adjust learning rates based on convergence"""
        if len(self.visual_usage_history) < 20:
            return
        
        # Check if we're consistently over or under budget
        recent_visual = list(self.visual_usage_history)[-20:]
        recent_text = list(self.text_usage_history)[-20:]
        
        visual_mean = np.mean(recent_visual)
        text_mean = np.mean(recent_text)
        
        visual_std = np.std(recent_visual)
        text_std = np.std(recent_text)
        
        # Increase learning rate if far from target
        visual_error = abs(visual_mean - self.visual_budget)
        text_error = abs(text_mean - self.text_budget)
        
        if visual_error > 0.2 * self.visual_budget:
            self.lambda_v_lr *= 1.1
        elif visual_std < 0.1 * self.visual_budget:
            self.lambda_v_lr *= 0.9
        
        if text_error > 0.2 * self.text_budget:
            self.lambda_t_lr *= 1.1
        elif text_std < 0.1 * self.text_budget:
            self.lambda_t_lr *= 0.9
        
        # Clip learning rates
        self.lambda_v_lr = np.clip(self.lambda_v_lr, 1e-4, 1e-2)
        self.lambda_t_lr = np.clip(self.lambda_t_lr, 1e-4, 1e-2)
    
    def get_multipliers(self) -> Tuple[float, float]:
        """Get current dual variable values"""
        return self.lambda_v, self.lambda_t
    
    def get_statistics(self) -> Dict:
        """Get statistics about budget usage"""
        stats = {
            "lambda_v": self.lambda_v,
            "lambda_t": self.lambda_t,
            "total_episodes": self.total_episodes,
            "visual_violations": self.budget_violations["visual"],
            "text_violations": self.budget_violations["text"],
            "violation_rate_visual": self.budget_violations["visual"] / max(1, self.total_episodes),
            "violation_rate_text": self.budget_violations["text"] / max(1, self.total_episodes)
        }
        
        if len(self.visual_usage_history) > 0:
            stats["avg_visual_usage"] = np.mean(list(self.visual_usage_history))
            stats["std_visual_usage"] = np.std(list(self.visual_usage_history))
        
        if len(self.text_usage_history) > 0:
            stats["avg_text_usage"] = np.mean(list(self.text_usage_history))
            stats["std_text_usage"] = np.std(list(self.text_usage_history))
        
        return stats


class OptimalStopping:
    """Implements optimal stopping decision for PDB"""
    
    def __init__(
        self,
        config: dict,
        cost_model: CostModel,
        gain_estimator: CombinedGainEstimator,
        dual_manager: DualVariableManager
    ):
        self.config = config
        self.cost_model = cost_model
        self.gain_estimator = gain_estimator
        self.dual_manager = dual_manager
        
        self.stopping_threshold = config.get("stopping_threshold", 0.0)
    
    def should_stop(
        self,
        actions: List[Action],
        state_embedding: torch.Tensor,
        state_dict: dict,
        history: List[Action] = None,
        budget_tracker: BudgetTracker = None
    ) -> Tuple[bool, Optional[Action], Dict]:
        """
        Determine whether to stop or which action to take
        
        Returns:
            should_stop: Whether to stop
            best_action: Best action if not stopping
            info: Dictionary with decision information
        """
        lambda_v, lambda_t = self.dual_manager.get_multipliers()
        
        best_action = None
        best_value = -float('inf')
        action_values = []
        
        for action in actions:
            # Skip if action exceeds budget
            if budget_tracker:
                visual_cost, text_cost = self.cost_model.compute_cost(action, state_dict)
                if not budget_tracker.can_afford(visual_cost, text_cost):
                    continue
            
            # Compute gain
            gain = self.gain_estimator.forward(
                action, state_embedding, state_dict, history
            )
            
            # Compute cost
            visual_cost, text_cost = self.cost_model.compute_cost(action, state_dict)
            
            # Compute value: gain - λ_v * visual_cost - λ_t * text_cost
            value = gain - lambda_v * visual_cost - lambda_t * text_cost
            
            action_values.append({
                "action": action,
                "gain": gain.item() if torch.is_tensor(gain) else gain,
                "visual_cost": visual_cost,
                "text_cost": text_cost,
                "value": value.item() if torch.is_tensor(value) else value
            })
            
            if value > best_value:
                best_value = value
                best_action = action
        
        # Decision: stop if best value is below threshold
        should_stop = best_value <= self.stopping_threshold
        
        # If stopping is not an explicit action, ensure we return stop signal
        if should_stop:
            best_action = StopAction()
        
        info = {
            "action_values": action_values,
            "best_value": best_value,
            "lambda_v": lambda_v,
            "lambda_t": lambda_t,
            "stopping_threshold": self.stopping_threshold
        }
        
        return should_stop, best_action, info


class PDBController:
    """Main controller for Primal-Dual Budgeter"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize components
        self.cost_model = CostModel(config)
        self.gain_estimator = CombinedGainEstimator(config)
        self.dual_manager = DualVariableManager(config)
        self.stopping = OptimalStopping(
            config, self.cost_model, self.gain_estimator, self.dual_manager
        )
    
    def initialize_episode(self) -> BudgetTracker:
        """Initialize a new episode"""
        return BudgetTracker(
            self.dual_manager.visual_budget,
            self.dual_manager.text_budget
        )
    
    def select_action(
        self,
        actions: List[Action],
        state_embedding: torch.Tensor,
        state_dict: dict,
        budget_tracker: BudgetTracker,
        history: List[Action] = None
    ) -> Tuple[Optional[Action], Dict]:
        """
        Select the next action or decide to stop
        
        Returns:
            action: Selected action (None if should stop)
            info: Decision information
        """
        should_stop, action, info = self.stopping.should_stop(
            actions, state_embedding, state_dict, history, budget_tracker
        )
        
        if should_stop:
            return None, info
        
        # Update budget tracker
        if action and not isinstance(action, StopAction):
            visual_cost, text_cost = self.cost_model.compute_cost(action, state_dict)
            budget_tracker.consume(action, visual_cost, text_cost)
        
        return action, info
    
    def end_episode(self, budget_tracker: BudgetTracker):
        """End an episode and update dual variables"""
        self.dual_manager.update(
            budget_tracker.visual_used,
            budget_tracker.text_used
        )
    
    def get_statistics(self) -> Dict:
        """Get controller statistics"""
        return {
            "dual_stats": self.dual_manager.get_statistics(),
            "config": self.config
        }