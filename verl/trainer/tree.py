import numpy as np
import torch
from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from ..utils.torch_functional import pad_sequence_to_length


class TreeNode:
    """
    TreeNode class for TreeGRPO algorithm.
    
    Each node represents a partial response at a specific depth in the generation tree.
    Nodes contain batch data, depth information, rewards, and connections to child nodes.
    """
    
    def __init__(self, batch_dict={}, depth=0, prompt_length=512, max_response_length=1280, 
                 step_length=256, reward=0, eos_token_id=None):
        self.depth = depth
        self.prompt_length = prompt_length
        self.max_response_length = max_response_length
        self.step_length = step_length
        
        self.E_reward = reward  # Expected reward for this node
        self.children = []
        self.child_rewards = []
        self.bro_rewards = []  # Brother node rewards (same depth, same parent)
        
        self.batch_dict = batch_dict
        self.eos_token_id = eos_token_id
        self.is_end = False
        
        # Validate batch dimensions
        if batch_dict and "input_ids" in batch_dict:
            assert self.batch_dict["input_ids"].shape[1] == self.prompt_length + self.depth * self.step_length
            assert self.batch_dict["input_ids"].shape == self.batch_dict["attention_mask"].shape
            # position_ids might have different dimensions for models like qwen2vl (3D instead of 2D)
            if "position_ids" in self.batch_dict:
                assert self.batch_dict["position_ids"].shape[-1] == self.batch_dict["input_ids"].shape[-1]
            assert self.batch_dict["input_ids"].shape[0] == 1
            
            # Check if generation has ended
            if depth != 0 and "responses" in self.batch_dict:
                valid_response_length = self.batch_dict["attention_mask"][0, -self.step_length:].sum().item()
                if valid_response_length > 0:
                    is_eos = self.batch_dict["responses"][0, valid_response_length - 1] == self.eos_token_id
                    self.is_end = is_eos.item() if torch.is_tensor(is_eos) else is_eos
                else:
                    self.is_end = False  # No valid response yet
    
    def add_child(self, child_node):
        """Add a child node to this node."""
        self.children.append(child_node)
    
    def format_node_input_batch(self):
        """Format the node's batch data for input to the model."""
        # Separate tensor and non-tensor data
        tensor_dict = {
            'input_ids': self.batch_dict['input_ids'],
            'attention_mask': self.batch_dict['attention_mask'],
        }
        
        # Add position_ids if present
        if 'position_ids' in self.batch_dict:
            tensor_dict['position_ids'] = self.batch_dict['position_ids']
        
        non_tensor_dict = {}
        # Only add non-tensor fields that exist in the batch_dict
        for key in ['data_source', 'ability', 'reward_model', 'extra_info', 'index', 'raw_prompt_ids']:
            if key in self.batch_dict:
                non_tensor_dict[key] = np.array(self.batch_dict[key])
        
        # Merge tensors and non-tensors
        all_data = {**tensor_dict, **non_tensor_dict}
        node_input_batch: DataProto = DataProto.from_single_dict(all_data)
        return node_input_batch
    
    def format_node_train_batch(self, pad_token_id):
        """Format the node's batch data for training, with proper padding."""
        max_depth = self.max_response_length // self.step_length
        assert self.max_response_length % self.step_length == 0
        left_pad_max_length = self.prompt_length + max_depth * self.step_length
        
        # Pad sequences to maximum length for training
        tensor_dict = {
            'input_ids': pad_sequence_to_length(
                self.batch_dict['input_ids'][0], left_pad_max_length, pad_token_id, left_pad=True
            ).unsqueeze(0),
            'attention_mask': pad_sequence_to_length(
                self.batch_dict['attention_mask'][0], left_pad_max_length, 0, left_pad=True
            ).unsqueeze(0),
            'prompts': pad_sequence_to_length(
                self.batch_dict['prompts'][0], left_pad_max_length-self.step_length, pad_token_id, left_pad=True
            ).unsqueeze(0),
            'responses': self.batch_dict['responses'],
            'token_level_rewards': self.batch_dict.get('token_level_rewards', torch.zeros_like(self.batch_dict['responses'])),
            'token_level_scores': self.batch_dict.get('token_level_scores', torch.zeros_like(self.batch_dict['responses'])),
            'advantages': self.batch_dict.get('advantages', torch.zeros_like(self.batch_dict['responses'])),
            'returns': self.batch_dict.get('returns', torch.zeros_like(self.batch_dict['responses'])),
        }
        
        # Handle position_ids separately due to potential 3D shape for models like qwen2vl
        if 'position_ids' in self.batch_dict:
            pos_ids = self.batch_dict['position_ids']
            if pos_ids.dim() == 2:  # Standard 2D position_ids
                tensor_dict['position_ids'] = pad_sequence_to_length(
                    pos_ids[0], left_pad_max_length, 0, left_pad=True
                ).unsqueeze(0)
            elif pos_ids.dim() == 3:  # 3D position_ids (e.g., qwen2vl with mrope)
                # Pad each dimension separately
                padded_pos_ids = []
                for i in range(pos_ids.shape[1]):
                    padded = pad_sequence_to_length(
                        pos_ids[0, i], left_pad_max_length, 0, left_pad=True
                    )
                    padded_pos_ids.append(padded)
                tensor_dict['position_ids'] = torch.stack(padded_pos_ids).unsqueeze(0)
        
        non_tensor_dict = {}
        # Only add non-tensor fields that exist in the batch_dict
        for key in ['data_source', 'ability', 'reward_model', 'extra_info', 'index', 'raw_prompt_ids']:
            if key in self.batch_dict:
                non_tensor_dict[key] = np.array(self.batch_dict[key])
        
        all_data = {**tensor_dict, **non_tensor_dict}
        node_input_batch: DataProto = DataProto.from_single_dict(all_data)
        return node_input_batch
    
    def print_node(self):
        """Print node information for debugging."""
        print(f"Node depth: {self.depth}")
        print(f"Node is_end: {self.is_end}")
        print(f"Node children: {len(self.children)}")
        print(f"Node E_reward: {self.E_reward}")
        print(f"Node batch_dict keys: {list(self.batch_dict.keys()) if self.batch_dict else 'None'}")