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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .core_algos import AdvantageEstimator, FixedKLController, KLController, compute_kl, get_kl_controller
from .simple_tree_trainer import SimpleTreeGRPOMixin
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_cgsg_contrastive_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    cgsg_labels: torch.Tensor,
    num_groups: int,
    samples_per_group: int,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute contrastive advantages for CGSG.
    
    Args:
        token_level_rewards: Token-level rewards (batch_size, seq_len)
        response_mask: Response mask (batch_size, seq_len)
        cgsg_labels: Labels indicating golden (1) or negative (0) samples
        num_groups: Number of prompt groups
        samples_per_group: Number of samples per group (1 golden + N negatives)
        eps: Small value for numerical stability
        
    Returns:
        advantages: Contrastive advantages
        returns: Returns (same as advantages for outcome-based rewards)
    """
    # Sum token rewards to get response-level rewards
    scores = token_level_rewards.sum(dim=-1)
    
    # Reshape to group structure
    scores = scores.view(num_groups, samples_per_group)
    labels = cgsg_labels.view(num_groups, samples_per_group)
    
    # Compute contrastive advantages
    advantages = torch.zeros_like(scores)
    
    for i in range(num_groups):
        group_scores = scores[i]
        group_labels = labels[i]
        
        # Find golden sample (label = 1)
        golden_mask = group_labels == 1
        golden_idx = golden_mask.nonzero(as_tuple=True)[0]
        
        if len(golden_idx) == 0:
            continue  # Skip if no golden sample
            
        # Find negative samples (label = 0)
        negative_mask = group_labels == 0
        negative_idx = negative_mask.nonzero(as_tuple=True)[0]
        
        if len(negative_idx) == 0:
            # Only golden sample, give it advantage of 1
            advantages[i, golden_idx[0]] = 1.0
            continue
        
        # Compute stable softmax using log-sum-exp trick for numerical stability
        max_score = group_scores.max()
        exp_scores = torch.exp(group_scores - max_score)
        
        # Compute log probabilities (which serve as initial advantages)
        log_probs = torch.log(exp_scores / (exp_scores.sum() + eps))
        
        # Use normalized advantages for more stable training
        # This helps prevent extreme gradients
        mean_log_prob = log_probs.mean()
        std_log_prob = log_probs.std() + eps
        normalized_advantages = (log_probs - mean_log_prob) / std_log_prob
        
        # Assign normalized advantages
        for j in range(len(group_scores)):
            advantages[i, j] = normalized_advantages[j].item()
    
    # Flatten back to batch dimension
    advantages = advantages.view(-1)
    
    # Expand to token level
    returns = advantages.unsqueeze(-1) * response_mask
    advantages = returns
    
    return advantages, returns


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    
    # Check if CGSG is applied
    if data.meta_info.get("cgsg_applied", False):
        loss_type = data.meta_info.get("loss_type", "normalized_advantage")
        cgsg_labels = data.batch["cgsg_labels"]
        
        if loss_type == "normalized_advantage":
            # Use standard GRPO advantage computation
            # The structure of the batch (golden + negatives) will naturally create contrast
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
        elif loss_type == "contrastive_loss":
            # Implement contrastive loss-based advantages
            advantages, returns = compute_cgsg_contrastive_advantage(
                token_level_rewards, response_mask, cgsg_labels, 
                data.meta_info["num_groups"], data.meta_info["samples_per_group"]
            )
        else:
            raise ValueError(f"Unknown CGSG loss type: {loss_type}")
    elif adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.TREE_GRPO:
        # For TreeGRPO, we'll handle advantage computation differently in the tree traversal
        # This is a placeholder that will be overridden by tree-specific computation
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer(SimpleTreeGRPOMixin):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    Inherits from SimpleTreeGRPOMixin to support TreeGRPO algorithm with simplified implementation.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")
        
        # TreeGRPO parameters are handled in the simplified implementation

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
    
    def _apply_pge_variant(self, batch: DataProto) -> DataProto:
        """Apply Perturbed Golden Ensemble (PGE) variant processing.
        
        1. Select golden samples (highest reward per prompt)
        2. Generate perturbations of golden samples
        3. Construct new batch with golden + perturbations
        4. Re-evaluate rewards for the new batch
        """
        from ..utils.perturbations import substitute_tokens, delete_tokens, insert_tokens
        
        pge_config = self.config.algorithm.pge_config
        n_rollouts = self.config.worker.rollout.n
        
        # Get rewards if not already computed
        if "token_level_scores" not in batch.batch:
            reward_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(batch))
            batch.batch["token_level_scores"] = reward_tensor
        
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
        
        vocab_size = len(self.tokenizer)
        special_token_ids = self.tokenizer.all_special_ids if hasattr(self.tokenizer, 'all_special_ids') else []
        
        for i in range(batch_size):
            # Get rewards for this prompt's rollouts
            start_idx = i * n_rollouts
            end_idx = (i + 1) * n_rollouts
            prompt_rewards = response_rewards[start_idx:end_idx]
            
            # Select golden sample (highest reward)
            golden_idx_local = torch.argmax(prompt_rewards).item()
            golden_idx = start_idx + golden_idx_local
            
            # Extract golden sample
            golden_response = batch.batch["responses"][golden_idx]
            golden_prompt = batch.batch["prompts"][golden_idx]
            golden_input_ids = batch.batch["input_ids"][golden_idx]
            golden_attention_mask = batch.batch["attention_mask"][golden_idx]
            golden_position_ids = batch.batch["position_ids"][golden_idx]
            golden_response_mask = batch.batch["response_mask"][golden_idx]
            
            # Add golden sample to new batch
            new_responses.append(golden_response)
            new_prompts.append(golden_prompt)
            new_input_ids.append(golden_input_ids)
            new_attention_masks.append(golden_attention_mask)
            new_position_ids.append(golden_position_ids)
            new_response_masks.append(golden_response_mask)
            
            # Generate perturbations
            num_perturbations = pge_config["num_perturbations"]
            perturbation_methods = pge_config["perturbation_methods"]
            perturbation_strength = pge_config["perturbation_strength"]
            
            for _ in range(num_perturbations):
                # Apply perturbations to golden response
                perturbed_response = golden_response.tolist()
                
                for method in perturbation_methods:
                    if method == "token_substitute":
                        perturbed_response = substitute_tokens(
                            perturbed_response, perturbation_strength, vocab_size, special_token_ids
                        )
                    elif method == "token_delete":
                        perturbed_response = delete_tokens(
                            perturbed_response, perturbation_strength, special_token_ids
                        )
                    elif method == "token_insert":
                        perturbed_response = insert_tokens(
                            perturbed_response, perturbation_strength, None
                        )
                
                # Convert back to tensor and create full sequence
                perturbed_response_tensor = torch.tensor(perturbed_response, device=golden_response.device)
                
                # Reconstruct full input_ids
                perturbed_input_ids = torch.cat([golden_prompt, perturbed_response_tensor])
                
                # Create attention mask and position IDs for perturbed sequence
                perturbed_attention_mask = torch.ones_like(perturbed_input_ids)
                perturbed_position_ids = torch.arange(len(perturbed_input_ids), device=golden_position_ids.device)
                
                # Create response mask
                perturbed_response_mask = torch.zeros_like(perturbed_input_ids)
                perturbed_response_mask[len(golden_prompt):] = 1
                
                # Add to new batch
                new_responses.append(perturbed_response_tensor)
                new_prompts.append(golden_prompt)
                new_input_ids.append(perturbed_input_ids)
                new_attention_masks.append(perturbed_attention_mask)
                new_position_ids.append(perturbed_position_ids)
                new_response_masks.append(perturbed_response_mask)
        
        # Pad sequences to same length before stacking
        from ..utils import torch_functional as VF
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        # Pad responses
        padded_responses = VF.pad_2d_list_to_length([r.tolist() if isinstance(r, torch.Tensor) else r for r in new_responses], 
                                         pad_token_id, max_length=None)
        padded_prompts = VF.pad_2d_list_to_length([p.tolist() if isinstance(p, torch.Tensor) else p for p in new_prompts], 
                                       pad_token_id, max_length=None)
        padded_input_ids = VF.pad_2d_list_to_length([i.tolist() if isinstance(i, torch.Tensor) else i for i in new_input_ids], 
                                          pad_token_id, max_length=None)
        
        # Update masks and position IDs to match padded length
        max_len = padded_input_ids.size(1)
        padded_attention_masks = []
        padded_position_ids = []
        padded_response_masks = []
        
        for i, orig_len in enumerate([len(seq) for seq in new_input_ids]):
            # Attention mask: 1 for real tokens, 0 for padding
            mask = torch.zeros(max_len, device=new_attention_masks[0].device)
            mask[:orig_len] = 1
            padded_attention_masks.append(mask)
            
            # Position IDs
            pos_ids = torch.arange(max_len, device=new_position_ids[0].device)
            padded_position_ids.append(pos_ids)
            
            # Response mask
            resp_mask = torch.zeros(max_len, device=new_response_masks[0].device)
            prompt_len = len(new_prompts[i])
            resp_mask[prompt_len:orig_len] = 1
            padded_response_masks.append(resp_mask)
        
        # Create UIDs for new batch
        new_uids = []
        for i in range(batch_size):
            # Each group gets same UID for GRPO advantage computation
            group_uid = str(uuid.uuid4())
            for _ in range(1 + num_perturbations):
                new_uids.append(group_uid)
        
        # Create new non-tensor batch
        new_non_tensor_batch = batch.non_tensor_batch.copy()
        new_non_tensor_batch["uid"] = np.array(new_uids, dtype=object)
        
        # Create new batch with golden samples and perturbations
        new_batch = DataProto(
            batch=TensorDict({
                "responses": padded_responses.to(batch.batch["responses"].device),
                "prompts": padded_prompts.to(batch.batch["prompts"].device),
                "input_ids": padded_input_ids.to(batch.batch["input_ids"].device),
                "attention_mask": torch.stack(padded_attention_masks),
                "position_ids": torch.stack(padded_position_ids),
                "response_mask": torch.stack(padded_response_masks),
            }, batch_size=len(new_responses)),
            non_tensor_batch=new_non_tensor_batch,
            meta_info=batch.meta_info
        )
        
        # Update meta info to indicate PGE was applied
        new_batch.meta_info["pge_applied"] = True
        new_batch.meta_info["num_groups"] = batch_size
        new_batch.meta_info["samples_per_group"] = 1 + num_perturbations
        
        return new_batch
    
    def _apply_cgsg_variant(self, batch: DataProto) -> DataProto:
        """Apply Contrastive Golden Sample Group (CGSG) variant processing.
        
        1. Select golden samples (all samples with reward=1, or highest reward if none)
        2. Select hard negatives (samples with reward<1)
        3. Construct new batch with golden + negatives
        """
        cgsg_config = self.config.algorithm.cgsg_config
        n_rollouts = self.config.worker.rollout.n
        
        # Get rewards if not already computed
        if "token_level_scores" not in batch.batch:
            reward_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(batch))
            batch.batch["token_level_scores"] = reward_tensor
        
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
        cgsg_labels = []  # 1 for golden, 0 for negatives
        
        num_negatives = cgsg_config["num_negatives"]
        negative_selection = cgsg_config["negative_selection_strategy"]
        
        for i in range(batch_size):
            # Get rewards for this prompt's rollouts
            start_idx = i * n_rollouts
            end_idx = (i + 1) * n_rollouts
            prompt_rewards = response_rewards[start_idx:end_idx]
            
            # Identify golden samples: all with reward >= threshold, or highest if none
            reward_threshold = cgsg_config.get("reward_threshold", 0.99)  # Consider rewards >= threshold as correct
            golden_mask = prompt_rewards >= reward_threshold
            
            if golden_mask.any():
                # Use all samples with reward >= threshold as golden samples
                golden_indices_local = torch.where(golden_mask)[0].tolist()
            else:
                # No perfect samples, use the single highest reward as golden
                golden_idx_local = torch.argmax(prompt_rewards).item()
                golden_indices_local = [golden_idx_local]
            
            # Add all golden samples
            for golden_idx_local in golden_indices_local:
                golden_idx = start_idx + golden_idx_local
                new_responses.append(batch.batch["responses"][golden_idx])
                new_prompts.append(batch.batch["prompts"][golden_idx])
                new_input_ids.append(batch.batch["input_ids"][golden_idx])
                new_attention_masks.append(batch.batch["attention_mask"][golden_idx])
                new_position_ids.append(batch.batch["position_ids"][golden_idx])
                new_response_masks.append(batch.batch["response_mask"][golden_idx])
                cgsg_labels.append(1.0)  # Golden sample label
            
            # Select negative samples (only from samples with reward < threshold)
            if negative_selection == "lowest_reward":
                # Get indices of non-golden samples
                all_indices = list(range(len(prompt_rewards)))
                non_golden_indices = [idx for idx in all_indices if idx not in golden_indices_local]
                
                # Only select from samples with reward < threshold as negatives
                negative_candidates = []
                for idx in non_golden_indices:
                    if prompt_rewards[idx] < reward_threshold:
                        negative_candidates.append((idx, prompt_rewards[idx].item()))
                
                # Sort by reward and select lowest ones
                negative_candidates.sort(key=lambda x: x[1])
                negative_indices_local = [idx for idx, _ in negative_candidates[:num_negatives]]
            else:
                raise ValueError(f"Unknown negative selection strategy: {negative_selection}")
            
            # Add negative samples
            for neg_idx_local in negative_indices_local:
                neg_idx = start_idx + neg_idx_local
                new_responses.append(batch.batch["responses"][neg_idx])
                new_prompts.append(batch.batch["prompts"][neg_idx])
                new_input_ids.append(batch.batch["input_ids"][neg_idx])
                new_attention_masks.append(batch.batch["attention_mask"][neg_idx])
                new_position_ids.append(batch.batch["position_ids"][neg_idx])
                new_response_masks.append(batch.batch["response_mask"][neg_idx])
                cgsg_labels.append(0.0)  # Negative sample label
        
        # Create UIDs for new batch
        new_uids = []
        samples_per_group_list = []  # Track actual samples per group
        
        # Correctly track samples per group based on actual batch construction
        for i in range(batch_size):
            # Get rewards for this prompt's rollouts
            start_idx = i * n_rollouts
            end_idx = (i + 1) * n_rollouts
            prompt_rewards = response_rewards[start_idx:end_idx]
            
            # Count golden samples
            reward_threshold = cgsg_config.get("reward_threshold", 0.99)
            golden_mask = prompt_rewards >= reward_threshold
            
            if golden_mask.any():
                num_golden = golden_mask.sum().item()
            else:
                num_golden = 1  # Single highest reward sample
            
            # Count negative samples that were actually added
            if negative_selection == "lowest_reward":
                # Count samples with reward < threshold
                negative_mask = prompt_rewards < reward_threshold
                num_negative_candidates = negative_mask.sum().item()
                # Actual negatives is min of candidates and requested num_negatives
                num_actual_negatives = min(num_negative_candidates, num_negatives)
            else:
                num_actual_negatives = 0
            
            actual_samples = num_golden + num_actual_negatives
            samples_per_group_list.append(actual_samples)
            
            # Each group gets same UID for GRPO advantage computation
            group_uid = str(uuid.uuid4())
            for _ in range(actual_samples):
                new_uids.append(group_uid)
        
        # Create new non-tensor batch
        new_non_tensor_batch = batch.non_tensor_batch.copy()
        new_non_tensor_batch["uid"] = np.array(new_uids, dtype=object)
        
        # Create new batch with contrastive samples
        new_batch = DataProto(
            batch=TensorDict({
                "responses": torch.stack(new_responses),
                "prompts": torch.stack(new_prompts),
                "input_ids": torch.stack(new_input_ids),
                "attention_mask": torch.stack(new_attention_masks),
                "position_ids": torch.stack(new_position_ids),
                "response_mask": torch.stack(new_response_masks),
                "cgsg_labels": torch.tensor(cgsg_labels, device=batch.batch["responses"].device),
            }, batch_size=len(new_responses)),
            non_tensor_batch=new_non_tensor_batch,
            meta_info=batch.meta_info
        )
        
        # Update meta info
        new_batch.meta_info["cgsg_applied"] = True
        new_batch.meta_info["loss_type"] = cgsg_config["loss_type"]
        new_batch.meta_info["num_groups"] = batch_size
        # Use the most common samples_per_group for reshaping (should be consistent)
        new_batch.meta_info["samples_per_group"] = max(set(samples_per_group_list), key=samples_per_group_list.count)
        new_batch.meta_info["samples_per_group_list"] = samples_per_group_list
        
        return new_batch

    def _make_batch_data(self, metrics: Dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    continue  # Continue generating more batches
                else:
                    raise ValueError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                batch = batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]
                
                # Apply PGE or CGSG variant processing if configured
                if self.config.algorithm.grpo_variant == "pge":
                    batch = self._apply_pge_variant(batch)
                elif self.config.algorithm.grpo_variant == "cgsg":
                    batch = self._apply_cgsg_variant(batch)
                    
                return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # TreeGRPO uses a different training approach
                if self.config.algorithm.adv_estimator == AdvantageEstimator.TREE_GRPO:
                    with timer("tree_grpo", timing_raw):
                        # Get a batch of prompts
                        batch_dict = next(self.data_iterator, None)
                        if batch_dict is None:
                            self.data_iterator = iter(self.train_dataloader)
                            batch_dict = next(self.data_iterator)
                        
                        input_batch = DataProto.from_single_dict(batch_dict)
                        input_batch.meta_info.update({
                            "min_pixels": self.config.data.min_pixels,
                            "max_pixels": self.config.data.max_pixels,
                            "video_fps": self.config.data.video_fps,
                            "temperature": self.config.worker.rollout.temperature,
                            "top_p": self.config.worker.rollout.top_p,
                            "top_k": self.config.worker.rollout.top_k,
                            "n": self.config.worker.rollout.n,
                            "max_new_tokens": self.config.worker.rollout.step_length,
                        })
                        
                        # Use standard generation for TreeGRPO
                        self.actor_rollout_ref_wg.prepare_rollout_engine()
                        batch = self._make_batch_data(metrics=metrics)
                        self.actor_rollout_ref_wg.release_rollout_engine()
                        
                        # Balance batch and compute global tokens
                        self._balance_batch(batch, metrics=metrics)
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        
                        # Compute rewards for TreeGRPO
                        with timer("reward", timing_raw):
                            reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(batch))
                            batch.batch["token_level_scores"] = reward_tensor
                            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                            metrics.update(reward_metrics)
                        
                        # Compute old_log_probs
                        with timer("old", timing_raw):
                            old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                            batch = batch.union(old_log_probs)
                        
                        # Compute ref_log_probs if using reference policy
                        if self.use_reference_policy:
                            with timer("ref", timing_raw):
                                ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                                batch = batch.union(ref_log_probs)
                        
                        # Apply KL penalty if needed
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        
                        # Compute TreeGRPO advantages on generated sequences
                        batch, train_acc, response_tokens = self._compute_batch_tree_advantage_simple(batch)
                        
                        if batch is None:
                            print("TreeGRPO: No valid training data generated, skipping step")
                            continue
                        
                        metrics["train/accuracy"] = train_acc
                        metrics["train/response_tokens"] = response_tokens
                else:
                    # Original PPO/GRPO training
                    with timer("gen", timing_raw):
                        self.actor_rollout_ref_wg.prepare_rollout_engine()
                        batch = self._make_batch_data(metrics=metrics)
                        self.actor_rollout_ref_wg.release_rollout_engine()

                # TreeGRPO already has computed advantages, skip most preprocessing steps
                if self.config.algorithm.adv_estimator != AdvantageEstimator.TREE_GRPO:
                    # balance the number of valid tokens on each dp rank.
                    # NOTE: this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # compute reward
                    if "token_level_scores" not in batch.batch:
                        with timer("reward", timing_raw):
                            reward_ref = self.reward_fn.compute_reward.remote(batch)

                    # recompute old_log_probs
                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with timer("adv", timing_raw):
                        if "token_level_scores" not in batch.batch:
                            # get token level scores asynchronously
                            reward_tensor, reward_metrics = ray.get(reward_ref)
                            batch.batch["token_level_scores"] = reward_tensor
                            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                            metrics.update(reward_metrics)

                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )
                else:
                    # TreeGRPO processing
                    self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    
                    # Compute rewards for TreeGRPO
                    with timer("reward", timing_raw):
                        reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(batch))
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)
                    
                    # recompute old_log_probs
                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)
                    
                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)
                    
                    # Apply KL penalty if needed
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    
                    # Compute TreeGRPO advantages
                    with timer("adv", timing_raw):
                        batch, train_acc, response_tokens = self._compute_batch_tree_advantage_simple(batch)
                        metrics.update({"train/accuracy": train_acc, "train/response_tokens": response_tokens})

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
