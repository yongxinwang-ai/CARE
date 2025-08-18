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
PDB (Primal-Dual Budgeter) information gain estimators
"""

from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .pdb_actions import Action, ActionType, CropAction, DrawAction, ToolAction


class GainEstimator:
    """Base class for information gain estimation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.temperature = config.get("gain_temperature", 1.0)
    
    def estimate_gain(
        self,
        action: Action,
        state: dict,
        history: List[Action] = None
    ) -> float:
        """Estimate information gain for an action"""
        raise NotImplementedError


class EntropyGainEstimator(GainEstimator):
    """Estimates gain based on entropy reduction"""
    
    def estimate_gain(
        self,
        action: Action,
        state: dict,
        history: List[Action] = None
    ) -> float:
        """
        Estimate gain as expected entropy reduction
        """
        current_entropy = state.get("answer_entropy", 1.0)
        
        # Estimate entropy after action
        if action.action_type == ActionType.ANS:
            # Direct answer reduces entropy to near zero
            expected_entropy = 0.1
        elif action.action_type == ActionType.TEXT:
            # Text generation reduces entropy based on tokens
            reduction_rate = 0.05  # per token
            expected_entropy = current_entropy * (1 - reduction_rate * action.num_tokens)
        elif action.action_type == ActionType.CROP:
            # Cropping can significantly reduce entropy if focusing on relevant region
            crop_relevance = self._estimate_crop_relevance(action, state)
            expected_entropy = current_entropy * (1 - 0.3 * crop_relevance)
        elif action.action_type == ActionType.DRAW:
            # Drawing helps with geometric reasoning
            if "geometry_context" in state:
                expected_entropy = current_entropy * 0.7
            else:
                expected_entropy = current_entropy * 0.95
        elif action.action_type == ActionType.TOOL:
            # Tools provide specific information
            if action.tool.name == "OCR":
                expected_entropy = current_entropy * 0.5
            elif action.tool.name == "DETECT":
                expected_entropy = current_entropy * 0.6
            else:
                expected_entropy = current_entropy * 0.7
        else:
            expected_entropy = current_entropy
        
        # Apply diminishing returns if action was recently used
        if history:
            recent_similar = sum(1 for a in history[-3:] 
                               if a.action_type == action.action_type)
            expected_entropy *= (1 + 0.1 * recent_similar)
        
        gain = max(0, current_entropy - expected_entropy)
        return gain / self.temperature
    
    def _estimate_crop_relevance(self, action: CropAction, state: dict) -> float:
        """Estimate relevance of crop region"""
        if "attention_map" in state:
            # Use attention to estimate relevance
            x1, y1, x2, y2 = action.bbox
            # Simplified: assume higher attention in center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            distance_from_center = ((center_x - 0.5)**2 + (center_y - 0.5)**2)**0.5
            relevance = 1.0 - distance_from_center
        else:
            relevance = 0.5  # Default
        return relevance


class ConsistencyGainEstimator(GainEstimator):
    """Estimates gain based on improving answer consistency"""
    
    def estimate_gain(
        self,
        action: Action,
        state: dict,
        history: List[Action] = None
    ) -> float:
        """
        Estimate gain as expected consistency improvement
        """
        current_consistency = state.get("answer_consistency", 0.5)
        
        # Estimate consistency improvement
        if action.action_type == ActionType.TEXT:
            # More reasoning improves consistency
            expected_consistency = min(1.0, current_consistency + 0.1)
        elif action.action_type == ActionType.CROP:
            # Focusing on details improves consistency
            expected_consistency = min(1.0, current_consistency + 0.15)
        elif action.action_type == ActionType.DRAW:
            # Visual aids improve geometric reasoning consistency
            if "geometry_context" in state:
                expected_consistency = min(1.0, current_consistency + 0.2)
            else:
                expected_consistency = current_consistency
        elif action.action_type == ActionType.TOOL:
            # Tools provide ground truth
            expected_consistency = min(1.0, current_consistency + 0.25)
        else:
            expected_consistency = current_consistency
        
        gain = max(0, expected_consistency - current_consistency)
        return gain / self.temperature


class TaskGainEstimator(GainEstimator):
    """Estimates gain based on task-specific features"""
    
    def estimate_gain(
        self,
        action: Action,
        state: dict,
        history: List[Action] = None
    ) -> float:
        """
        Estimate gain based on task requirements
        """
        task_type = state.get("task_type", "general")
        
        gain = 0.0
        
        if task_type == "geometry":
            if action.action_type == ActionType.DRAW:
                # Drawing is highly valuable for geometry
                gain = 0.4
                if action.primitive.name in ["ANGLE", "MEASURE", "PARALLEL"]:
                    gain = 0.6
            elif action.action_type == ActionType.CROP:
                # Zooming on geometric figures
                gain = 0.3
        
        elif task_type == "text_reading":
            if action.action_type == ActionType.TOOL and action.tool.name == "OCR":
                gain = 0.7
            elif action.action_type == ActionType.CROP:
                # Zoom on text regions
                gain = 0.3
        
        elif task_type == "counting":
            if action.action_type == ActionType.TOOL and action.tool.name == "DETECT":
                gain = 0.6
            elif action.action_type == ActionType.CROP:
                gain = 0.2
        
        elif task_type == "table":
            if action.action_type == ActionType.TOOL and action.tool.name == "TABLE":
                gain = 0.8
        
        # General gains
        if gain == 0.0:
            if action.action_type == ActionType.TEXT:
                gain = 0.2
            elif action.action_type == ActionType.CROP:
                gain = 0.15
        
        return gain / self.temperature


class CombinedGainEstimator(nn.Module):
    """Neural network that combines multiple gain estimators"""
    
    def __init__(self, config: dict, hidden_dim: int = 256):
        super().__init__()
        self.config = config
        
        # Initialize individual estimators
        self.entropy_estimator = EntropyGainEstimator(config)
        self.consistency_estimator = ConsistencyGainEstimator(config)
        self.task_estimator = TaskGainEstimator(config)
        
        # Get weights from config
        estimator_names = config.get("gain_estimators", ["entropy", "consistency", "task"])
        weights = config.get("gain_weights", [0.4, 0.3, 0.3])
        self.estimator_weights = dict(zip(estimator_names, weights))
        
        # Neural network for learned gain refinement
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, 128),  # +3 for the three base gains
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Calibration layer
        self.calibration = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(
        self,
        action: Action,
        state_embedding: torch.Tensor,
        state_dict: dict,
        history: List[Action] = None
    ) -> torch.Tensor:
        """
        Forward pass to compute combined gain
        """
        # Compute base gains
        entropy_gain = self.entropy_estimator.estimate_gain(action, state_dict, history)
        consistency_gain = self.consistency_estimator.estimate_gain(action, state_dict, history)
        task_gain = self.task_estimator.estimate_gain(action, state_dict, history)
        
        # Weighted combination
        base_gain = (
            self.estimator_weights.get("entropy", 0.4) * entropy_gain +
            self.estimator_weights.get("consistency", 0.3) * consistency_gain +
            self.estimator_weights.get("task", 0.3) * task_gain
        )
        
        # Neural refinement
        gain_features = torch.tensor([entropy_gain, consistency_gain, task_gain])
        combined_features = torch.cat([state_embedding, gain_features])
        refined_gain = self.gain_head(combined_features).squeeze()
        
        # Combine base and refined
        final_gain = 0.5 * base_gain + 0.5 * refined_gain
        
        # Calibration
        calibrated_gain = self.calibration(final_gain.unsqueeze(-1)).squeeze()
        
        return calibrated_gain
    
    def estimate_batch(
        self,
        actions: List[Action],
        state_embeddings: torch.Tensor,
        state_dicts: List[dict],
        histories: List[List[Action]] = None
    ) -> torch.Tensor:
        """
        Batch estimation of gains
        """
        gains = []
        for i, action in enumerate(actions):
            state_emb = state_embeddings[i] if len(state_embeddings.shape) > 1 else state_embeddings
            state_dict = state_dicts[i] if isinstance(state_dicts, list) else state_dicts
            history = histories[i] if histories else None
            
            gain = self.forward(action, state_emb, state_dict, history)
            gains.append(gain)
        
        return torch.stack(gains)


class GainCalibrator:
    """Calibrates gain estimates using historical data"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = []
        self.quantiles = None
    
    def update(self, predicted_gain: float, actual_gain: float):
        """Update calibration with new observation"""
        self.history.append((predicted_gain, actual_gain))
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Update quantiles
        if len(self.history) >= 10:
            predicted = np.array([h[0] for h in self.history])
            actual = np.array([h[1] for h in self.history])
            
            # Compute calibration mapping
            percentiles = np.linspace(0, 100, 11)
            self.quantiles = {
                "predicted": np.percentile(predicted, percentiles),
                "actual": np.percentile(actual, percentiles)
            }
    
    def calibrate(self, gain: float) -> float:
        """Apply calibration to gain estimate"""
        if self.quantiles is None:
            return gain
        
        # Find nearest quantile and interpolate
        pred_q = self.quantiles["predicted"]
        actual_q = self.quantiles["actual"]
        
        # Linear interpolation
        idx = np.searchsorted(pred_q, gain)
        if idx == 0:
            return actual_q[0]
        elif idx >= len(pred_q):
            return actual_q[-1]
        else:
            # Interpolate
            t = (gain - pred_q[idx-1]) / (pred_q[idx] - pred_q[idx-1])
            return actual_q[idx-1] + t * (actual_q[idx] - actual_q[idx-1])