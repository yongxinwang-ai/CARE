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
PDB (Primal-Dual Budgeter) cost models for actions
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn

from .pdb_actions import (
    Action, ActionType, AnswerAction, TextAction, CropAction,
    DrawAction, ToolAction, StopAction, DrawPrimitive, ToolType
)


class CostModel:
    """Base class for computing action costs"""
    
    def __init__(self, config: dict):
        self.config = config
        self.cost_per_patch = config.get("cost_per_patch", 1)
        self.cost_crop_fixed = config.get("cost_crop_fixed", 32)
        self.cost_draw_per_primitive = config.get("cost_draw_per_primitive", 4)
        self.cost_per_text_token = config.get("cost_per_text_token", 1)
        self.cost_tool_ocr = config.get("cost_tool_ocr", 64)
        self.cost_tool_detect = config.get("cost_tool_detect", 48)
    
    def compute_cost(
        self,
        action: Action,
        state: dict = None
    ) -> Tuple[float, float]:
        """
        Compute visual and text costs for an action
        Returns: (visual_cost, text_cost)
        """
        if isinstance(action, AnswerAction):
            return self._compute_answer_cost(state)
        elif isinstance(action, TextAction):
            return self._compute_text_cost(action, state)
        elif isinstance(action, CropAction):
            return self._compute_crop_cost(action, state)
        elif isinstance(action, DrawAction):
            return self._compute_draw_cost(action, state)
        elif isinstance(action, ToolAction):
            return self._compute_tool_cost(action, state)
        elif isinstance(action, StopAction):
            return (0.0, 0.0)
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
    
    def _compute_answer_cost(self, state: dict) -> Tuple[float, float]:
        """Cost for direct answer action"""
        # Estimate answer length based on context
        estimated_tokens = 10  # Default estimate
        if state and "answer_type" in state:
            if state["answer_type"] == "short":
                estimated_tokens = 5
            elif state["answer_type"] == "long":
                estimated_tokens = 20
        
        return (0.0, estimated_tokens * self.cost_per_text_token)
    
    def _compute_text_cost(self, action: TextAction, state: dict) -> Tuple[float, float]:
        """Cost for text generation action"""
        text_cost = action.num_tokens * self.cost_per_text_token
        return (0.0, text_cost)
    
    def _compute_crop_cost(self, action: CropAction, state: dict) -> Tuple[float, float]:
        """Cost for crop action"""
        # Fixed cost for cropping
        visual_cost = self.cost_crop_fixed
        
        # Additional cost for re-encoding patches
        x1, y1, x2, y2 = action.bbox
        crop_area = (x2 - x1) * (y2 - y1)
        
        # Estimate number of patches based on crop area
        # Assuming higher resolution after crop
        estimated_patches = int(crop_area * 256)  # 256 patches for full high-res
        visual_cost += estimated_patches * self.cost_per_patch
        
        return (visual_cost, 0.0)
    
    def _compute_draw_cost(self, action: DrawAction, state: dict) -> Tuple[float, float]:
        """Cost for drawing action"""
        # Base cost per primitive
        visual_cost = self.cost_draw_per_primitive
        
        # Additional cost based on primitive complexity
        if action.primitive in [DrawPrimitive.PARALLEL, DrawPrimitive.PERPENDICULAR]:
            visual_cost *= 2  # Multiple lines
        elif action.primitive == DrawPrimitive.ANGLE:
            visual_cost *= 1.5  # Angle measurement
        elif action.primitive == DrawPrimitive.MEASURE:
            visual_cost *= 1.2  # Distance measurement
        
        # If rendering is required (not just vector output)
        if state and state.get("render_draw", False):
            visual_cost += 16 * self.cost_per_patch  # Low-res render
        
        return (visual_cost, 0.0)
    
    def _compute_tool_cost(self, action: ToolAction, state: dict) -> Tuple[float, float]:
        """Cost for tool action"""
        if action.tool == ToolType.OCR:
            visual_cost = self.cost_tool_ocr
        elif action.tool == ToolType.DETECT:
            visual_cost = self.cost_tool_detect
        elif action.tool == ToolType.TABLE:
            visual_cost = self.cost_tool_ocr * 1.5  # Table extraction is more expensive
        elif action.tool == ToolType.FORMULA:
            visual_cost = self.cost_tool_ocr * 1.2
        else:
            visual_cost = self.cost_tool_ocr  # Default
        
        # Add region-specific cost if applicable
        if action.region is not None:
            x1, y1, x2, y2 = action.region
            region_area = (x2 - x1) * (y2 - y1)
            visual_cost *= region_area  # Scale by region size
        
        return (visual_cost, 0.0)


class LearnedCostModel(nn.Module):
    """Neural network-based cost model that can be trained"""
    
    def __init__(self, config: dict, hidden_dim: int = 256):
        super().__init__()
        self.config = config
        self.base_cost_model = CostModel(config)
        
        # Neural network for cost refinement
        self.visual_cost_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive costs
        )
        
        self.text_cost_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive costs
        )
    
    def forward(
        self,
        action: Action,
        state_embedding: torch.Tensor,
        state_dict: dict = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute costs
        Args:
            action: Action to evaluate
            state_embedding: Embedding of current state
            state_dict: Dictionary with state information
        Returns:
            (visual_cost, text_cost) as tensors
        """
        # Get base costs
        base_visual, base_text = self.base_cost_model.compute_cost(action, state_dict)
        
        # Refine with neural network
        visual_refinement = self.visual_cost_head(state_embedding).squeeze()
        text_refinement = self.text_cost_head(state_embedding).squeeze()
        
        # Combine base and learned costs
        visual_cost = torch.tensor(base_visual) + visual_refinement
        text_cost = torch.tensor(base_text) + text_refinement
        
        return visual_cost, text_cost


class BudgetTracker:
    """Tracks budget consumption during rollout"""
    
    def __init__(self, visual_budget: float, text_budget: float):
        self.visual_budget = visual_budget
        self.text_budget = text_budget
        self.visual_used = 0.0
        self.text_used = 0.0
        self.action_history = []
    
    def can_afford(self, visual_cost: float, text_cost: float) -> bool:
        """Check if action is within budget"""
        return (self.visual_used + visual_cost <= self.visual_budget and
                self.text_used + text_cost <= self.text_budget)
    
    def consume(self, action: Action, visual_cost: float, text_cost: float):
        """Consume budget for an action"""
        self.visual_used += visual_cost
        self.text_used += text_cost
        self.action_history.append({
            "action": action,
            "visual_cost": visual_cost,
            "text_cost": text_cost,
            "visual_total": self.visual_used,
            "text_total": self.text_used
        })
    
    def get_remaining(self) -> Tuple[float, float]:
        """Get remaining budget"""
        return (self.visual_budget - self.visual_used,
                self.text_budget - self.text_used)
    
    def get_usage_ratio(self) -> Tuple[float, float]:
        """Get budget usage ratio"""
        visual_ratio = self.visual_used / self.visual_budget if self.visual_budget > 0 else 0
        text_ratio = self.text_used / self.text_budget if self.text_budget > 0 else 0
        return (visual_ratio, text_ratio)