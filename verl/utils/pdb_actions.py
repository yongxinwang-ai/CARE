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
PDB (Primal-Dual Budgeter) action definitions for multimodal reasoning
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, Union
import torch


class ActionType(Enum):
    """Types of actions available in PDB"""
    ANS = auto()      # Direct answer
    TEXT = auto()     # Generate text/CoT tokens
    CROP = auto()     # Crop region of image
    DRAW = auto()     # Draw primitives on image
    TOOL = auto()     # Use external tools (OCR, detection, etc.)
    STOP = auto()     # Stop and return current answer


class DrawPrimitive(Enum):
    """Types of drawing primitives"""
    LINE = auto()          # Draw a line
    PARALLEL = auto()      # Draw parallel lines
    PERPENDICULAR = auto() # Draw perpendicular lines
    ANGLE = auto()         # Mark/measure angle
    MEASURE = auto()       # Measure distance
    CIRCLE = auto()        # Draw circle
    RECTANGLE = auto()     # Draw rectangle
    ARROW = auto()         # Draw arrow/vector


class ToolType(Enum):
    """Types of external tools"""
    OCR = auto()           # Optical character recognition
    DETECT = auto()        # Object detection
    TABLE = auto()         # Table extraction
    FORMULA = auto()       # Formula recognition


@dataclass
class Action:
    """Base action class"""
    action_type: ActionType
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnswerAction(Action):
    """Direct answer action"""
    def __init__(self):
        super().__init__(ActionType.ANS)


@dataclass
class TextAction(Action):
    """Generate text/CoT tokens"""
    num_tokens: int = 10  # Number of tokens to generate
    
    def __init__(self, num_tokens: int = 10):
        super().__init__(ActionType.TEXT)
        self.num_tokens = num_tokens


@dataclass
class CropAction(Action):
    """Crop a region of the image"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized coordinates
    
    def __init__(self, bbox: Tuple[float, float, float, float]):
        super().__init__(ActionType.CROP)
        self.bbox = bbox
        self.metadata = {"bbox": bbox}


@dataclass
class DrawAction(Action):
    """Draw primitives on image"""
    primitive: DrawPrimitive
    params: dict  # Parameters specific to the primitive
    
    def __init__(self, primitive: DrawPrimitive, params: dict):
        super().__init__(ActionType.DRAW)
        self.primitive = primitive
        self.params = params
        self.metadata = {"primitive": primitive.name, "params": params}


@dataclass
class ToolAction(Action):
    """Use external tool"""
    tool: ToolType
    region: Optional[Tuple[float, float, float, float]] = None  # Optional region to apply tool
    
    def __init__(self, tool: ToolType, region: Optional[Tuple[float, float, float, float]] = None):
        super().__init__(ActionType.TOOL)
        self.tool = tool
        self.region = region
        self.metadata = {"tool": tool.name, "region": region}


@dataclass
class StopAction(Action):
    """Stop and return current answer"""
    def __init__(self):
        super().__init__(ActionType.STOP)


class ActionSpace:
    """Manages the space of available actions"""
    
    def __init__(self, config: dict):
        self.config = config
        self.enable_crop = config.get("enable_crop", True)
        self.enable_draw = config.get("enable_draw", True)
        self.enable_tool = config.get("enable_tool", False)
        self.max_crop_candidates = config.get("max_crop_candidates", 5)
        self.max_draw_primitives = config.get("max_draw_primitives", 3)
    
    def get_available_actions(
        self,
        state: dict,
        image_features: Optional[torch.Tensor] = None
    ) -> List[Action]:
        """Get list of available actions given current state"""
        actions = []
        
        # Always available actions
        actions.append(AnswerAction())
        actions.append(StopAction())
        
        # Text generation actions
        for num_tokens in [5, 10, 20]:
            actions.append(TextAction(num_tokens))
        
        # Crop actions
        if self.enable_crop and image_features is not None:
            crop_candidates = self._generate_crop_candidates(state, image_features)
            actions.extend(crop_candidates[:self.max_crop_candidates])
        
        # Draw actions
        if self.enable_draw:
            draw_candidates = self._generate_draw_candidates(state)
            actions.extend(draw_candidates[:self.max_draw_primitives])
        
        # Tool actions
        if self.enable_tool:
            tool_candidates = self._generate_tool_candidates(state)
            actions.extend(tool_candidates)
        
        return actions
    
    def _generate_crop_candidates(
        self,
        state: dict,
        image_features: torch.Tensor
    ) -> List[CropAction]:
        """Generate candidate crop regions"""
        candidates = []
        
        # Generate grid-based crops
        grid_sizes = [(2, 2), (3, 3)]
        for rows, cols in grid_sizes:
            for i in range(rows):
                for j in range(cols):
                    x1 = j / cols
                    y1 = i / rows
                    x2 = (j + 1) / cols
                    y2 = (i + 1) / rows
                    candidates.append(CropAction((x1, y1, x2, y2)))
        
        # Add attention-based crops if available
        if "attention_map" in state:
            # Extract high-attention regions
            pass
        
        return candidates
    
    def _generate_draw_candidates(self, state: dict) -> List[DrawAction]:
        """Generate candidate drawing actions"""
        candidates = []
        
        # Generate basic drawing primitives
        if "geometry_context" in state:
            # Add geometry-specific primitives
            candidates.append(DrawAction(
                DrawPrimitive.ANGLE,
                {"vertices": [(0.3, 0.3), (0.5, 0.5), (0.7, 0.3)]}
            ))
            candidates.append(DrawAction(
                DrawPrimitive.PARALLEL,
                {"lines": [((0.2, 0.3), (0.8, 0.3)), ((0.2, 0.6), (0.8, 0.6))]}
            ))
        
        # Add measurement primitives
        candidates.append(DrawAction(
            DrawPrimitive.MEASURE,
            {"points": [(0.2, 0.5), (0.8, 0.5)]}
        ))
        
        return candidates
    
    def _generate_tool_candidates(self, state: dict) -> List[ToolAction]:
        """Generate candidate tool actions"""
        candidates = []
        
        # OCR tool
        if "text_regions" in state:
            candidates.append(ToolAction(ToolType.OCR))
        
        # Detection tool
        if "objects_needed" in state:
            candidates.append(ToolAction(ToolType.DETECT))
        
        # Table tool
        if "table_present" in state:
            candidates.append(ToolAction(ToolType.TABLE))
        
        return candidates