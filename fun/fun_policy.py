import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_ops import sequence_mask

from optim.RMSpropLambdaLR import RMSpropLambdaLR
from optim.RMSpropCyclicLR import RMSpropCyclicLR

# Merge modified config with A2C and A3C default config
FUN_CONFIG = A2CTrainer.merge_trainer_configs(
    A2C_DEFAULT_CONFIG,
    {
        'use_gae': False,

        'lr': 1e-3,

        # Can be either constant, anneal, or cyclic
        'lr_mode': 'constant',

        # Linear learning rate annealing
        'end_lr': 1e-4,
        'anneal_timesteps': 10000000,

        # Cyclic learning rate
        'cyclic_lr_base_lr': 1e-4,
        'cyclic_lr_max_lr': 1e-3,
        'cyclic_lr_step_size': 200,
        'cyclic_lr_mode': 'triangular',
        'cyclic_lr_gamma': 0.99,

        'grad_clip': 0.5,
        'epsilon': 1e-8,

        '_use_trajectory_view_api': False,
    },
    _allow_unknown_configs=True,
)

def model_extra_out(policy, input_dict, state_batches, model, action_dist):
    """
    Collects additional output for batches of experiences / observations.
    """
    # Get the manager and worker value estimations
    manager_values, worker_values = model.value_function()

    # Manager latent state and goal have shape [Batch, Time, Features]
    # Time dimension is squeezed as it is always 1
    manager_latent_state, manager_goal = model.manager_features()
    manager_latent_state = torch.squeeze(manager_latent_state, 1)
    manager_goal = torch.squeeze(manager_goal, 1)

    return {
        'manager_values': manager_values,
        'worker_values': worker_values,
        'manager_latent_state': manager_latent_state,
        'manager_goal': manager_goal,
    }

def postprocesses_trajectories(
        policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Postprocesses individual trajectories.

    Inputs are numpy arrays with shape [Time, Feature Dims...] or [Time]
    if there is only one feature. Note that inputs are not batched.

    Computes advantages.
    """
    horizon = 5
    manager_latent_state = torch.Tensor(sample_batch['manager_latent_state'])
    manager_goal = torch.Tensor(sample_batch['manager_goal'])

    fun_intrinsic_reward = []
    for i in range(manager_latent_state.shape[0]):
        reward = 0
        reward_horizon = 0
        for j in range(1, horizon + 1):
            index = i - j
            if index >= 0:
                manager_latent_state_current = manager_latent_state[i]
                manager_latent_state_prev = manager_latent_state[index]
                manager_latent_state_diff = manager_latent_state_current - manager_latent_state_prev
                manager_goal_prev = manager_goal[index]
                reward = reward + F.cosine_similarity(manager_latent_state_diff, manager_goal_prev, dim=0)
                reward_horizon += 1
        if reward_horizon > 0:
            reward = reward / reward_horizon
        fun_intrinsic_reward.append(reward)

    fun_intrinsic_reward = np.array(fun_intrinsic_reward)
    sample_batch['fun_intrinsic_reward'] = fun_intrinsic_reward

    completed = sample_batch[SampleBatch.DONES][-1]
    if completed:
        manager_last_r = 0.0
        worker_last_r = 0.0
    else:
        # Trajectory has been truncated, estimate final reward using the
        # value function from the terminal observation and
        # internal recurrent state if any
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append(sample_batch['state_out_{}'.format(i)][-1])
        manager_last_r, worker_last_r = policy._value(
            sample_batch[SampleBatch.NEXT_OBS][-1],
            sample_batch[SampleBatch.ACTIONS][-1],
            sample_batch[SampleBatch.REWARDS][-1],
            *next_state)
        manager_last_r = manager_last_r[0]
        worker_last_r = worker_last_r[0]

    # Compute advantages and value targets for the manager
    sample_batch[SampleBatch.VF_PREDS] = sample_batch['manager_values']
    sample_batch = compute_advantages(
        sample_batch, manager_last_r, policy.config['gamma'],
        policy.config['lambda'], policy.config['use_gae'],
        policy.config['use_critic'])
    sample_batch['manager_advantages'] = sample_batch[Postprocessing.ADVANTAGES]
    sample_batch['manager_value_targets'] = sample_batch[Postprocessing.VALUE_TARGETS]

    sample_batch[SampleBatch.REWARDS] = (sample_batch[SampleBatch.REWARDS]
        + fun_intrinsic_reward)

    # Compute advantages and value targets for the worker
    sample_batch[SampleBatch.VF_PREDS] = sample_batch['worker_values']
    sample_batch = compute_advantages(
        sample_batch, worker_last_r, policy.config['gamma'],
        policy.config['lambda'], policy.config['use_gae'],
        policy.config['use_critic'])
    sample_batch['worker_advantages'] = sample_batch[Postprocessing.ADVANTAGES]
    sample_batch['worker_value_targets'] = sample_batch[Postprocessing.VALUE_TARGETS]

    # WARNING: These values are only used temporarily. Do not use:
    # sample_batch[SampleBatch.VF_PREDS]
    # sample_batch[Postprocessing.ADVANTAGES]
    # sample_batch[Postprocessing.VALUE_TARGETS]

    return sample_batch

# Modify losses to average over batches instead of sum
# Provides consistent behaviour when changing batch sizes
# Fix loss to mask padded sequences when training with recurrent policies
def actor_critic_loss(policy, model, dist_class, train_batch):
    assert policy.is_recurrent(), "policy must be recurrent"

    seq_lens = train_batch['seq_lens']
    max_seq_len = torch.max(seq_lens)
    mask_orig = sequence_mask(seq_lens, max_seq_len)
    mask = torch.reshape(mask_orig, [-1])

    logits, _ = model.from_batch(train_batch)
    manager_values, worker_values = model.value_function()

    manager_latent_state, manager_goal = model.manager_features()

    horizon = 5
    manager_latent_state_future = manager_latent_state[:, horizon:, :]
    manager_latent_state_future = F.pad(
        manager_latent_state_future,
        (0, 0, 0, horizon),
        'constant',
        0)
    manager_latent_state_diff = (manager_latent_state_future - manager_latent_state).detach()

    horizon_mask = mask_orig
    horizon_mask[:, -horizon:] = 0
    horizon_mask = horizon_mask.reshape(-1)

    policy.manager_loss = -torch.sum(
        train_batch['manager_advantages']
        * F.cosine_similarity(manager_latent_state_diff, manager_goal, dim=-1).reshape(-1)
        * horizon_mask)

    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS])
    policy.entropy = -torch.sum(dist.entropy() * mask)
    policy.pi_err = -torch.sum(train_batch['worker_advantages'] * log_probs.reshape(-1) * mask)

    policy.manager_value_err = torch.sum(torch.pow((manager_values.reshape(-1) - train_batch['manager_value_targets']) * mask, 2.0))
    policy.worker_value_err = torch.sum(torch.pow((worker_values.reshape(-1) - train_batch['worker_value_targets']) * mask, 2.0))

    overall_err = sum([
        policy.pi_err,
        policy.config['vf_loss_coeff'] * policy.manager_value_err,
        policy.config['vf_loss_coeff'] * policy.worker_value_err,
        policy.config['entropy_coeff'] * policy.entropy,
        policy.manager_loss,
    ])
    return overall_err

# Fix value network mixin to use internal recurrent state if any
class ValueNetworkMixin:
    def _value(self, obs, prev_action, prev_reward, *state):
        _ = self.model({
            SampleBatch.CUR_OBS: torch.Tensor([obs]).to(self.device),
            SampleBatch.PREV_ACTIONS: torch.Tensor([prev_action]).to(self.device),
            SampleBatch.PREV_REWARDS: torch.Tensor([prev_reward]).to(self.device),
        }, [torch.Tensor([s]).to(self.device) for s in state], torch.Tensor([1]).to(self.device))
        return self.model.value_function()

def torch_optimizer(policy, config):
    optimizers = {}
    optimizers['constant'] = torch_rmsprop_optimizer
    optimizers['anneal'] = torch_rmsprop_lambdalr_optimizer
    optimizers['cyclic'] = torch_rmsprop_cyclic_lr_optimizer
    optimizer = optimizers[config['lr_mode']]
    return optimizer(policy, config)

# Use RMSprop as per source paper
# More consistent than ADAM in non-stationary problems such as RL
def torch_rmsprop_optimizer(policy, config):
    return optim.RMSprop(
        policy.model.parameters(),
        lr=config['lr'],
        eps=config['epsilon'])

# RMSprop with linear learning rate annealing
def torch_rmsprop_lambdalr_optimizer(policy, config):
    if config['num_workers'] == 0:
        num_workers = 1
    else:
        num_workers = config['num_workers']

    batch_size = (num_workers
        * config['num_envs_per_worker']
        * config['rollout_fragment_length'])

    anneal_steps = float(config['anneal_timesteps']) / float(batch_size)

    lr = float(config['lr'])
    end_lr = float(config['end_lr'])

    return RMSpropLambdaLR(
        policy.model.parameters(),
        lr=config['lr'],
        eps=config['epsilon'],
        lr_lambda=lambda x: 1. - ((1. - (end_lr / lr)) * (x / anneal_steps)))

def torch_rmsprop_cyclic_lr_optimizer(policy, config):
    return RMSpropCyclicLR(
        policy.model.parameters(),
        lr=config['cyclic_lr_base_lr'],
        eps=config['epsilon'],
        base_lr=config['cyclic_lr_base_lr'],
        max_lr=config['cyclic_lr_max_lr'],
        step_size_up=config['cyclic_lr_step_size'],
        mode=config['cyclic_lr_mode'],
        gamma=config['cyclic_lr_gamma'])

# Update stats function to include the current learning rate
def stats(policy, train_batch):
    return {
        'policy_entropy': policy.entropy.item(),
        'policy_loss': policy.pi_err.item(),
        'manager_loss': policy.manager_loss.item(),
        'manager_vf_loss': policy.manager_value_err.item(),
        'worker_vf_loss': policy.worker_value_err.item(),
        'cur_lr': policy._optimizers[0].param_groups[0]['lr'],
        'fun_intrinsic_reward': train_batch['fun_intrinsic_reward'].mean().item()
    }

def get_policy_class(config):
    return FuNPolicy

FuNPolicy = A3CTorchPolicy.with_updates(
        name='FuNPolicy',
        get_default_config=lambda: FUN_CONFIG,
        extra_action_out_fn=model_extra_out,
        postprocess_fn=postprocesses_trajectories,
        loss_fn=actor_critic_loss,
        stats_fn=stats,
        mixins=[ValueNetworkMixin],
        optimizer_fn=torch_optimizer)

FuNTrainer = A2CTrainer.with_updates(
    name='FuN',
    default_config=FUN_CONFIG,
    default_policy=FuNPolicy,
    get_policy_class=get_policy_class)
