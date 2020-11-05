import torch
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
TUNED_A2C_CONFIG = A2CTrainer.merge_trainer_configs(
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

# Modify losses to average over batches instead of sum
# Provides consistent behaviour when changing batch sizes
# Fix loss to mask padded sequences when training with recurrent policies


def actor_critic_loss(policy, model, dist_class, train_batch):
    # If policy is recurrent, mask out padded sequences
    # and calculate batch size
    if policy.is_recurrent():
        seq_lens = train_batch['seq_lens']
        max_seq_len = torch.max(seq_lens)
        mask_orig = sequence_mask(seq_lens, max_seq_len)
        mask = torch.reshape(mask_orig, [-1])
        batch_size = seq_lens.shape[0]
    else:
        mask = torch.ones_like(train_batch[SampleBatch.REWARDS])
        batch_size = mask.shape[0]

    icm_loss, _ = model.icm_forward(
        train_batch[SampleBatch.OBS],
        train_batch[SampleBatch.NEXT_OBS],
        train_batch[SampleBatch.ACTIONS]
    )
    icm_loss = torch.sum(icm_loss)
    icm_loss /= batch_size
    policy.icm_loss = icm_loss

    logits, _ = model.from_batch(train_batch)
    values = model.value_function()
    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS])
    policy.entropy = -torch.sum(dist.entropy() * mask) / batch_size
    policy.pi_err = - \
        torch.sum(train_batch[Postprocessing.ADVANTAGES]
                  * log_probs.reshape(-1) * mask) / batch_size
    policy.value_err = torch.sum(torch.pow(
        (values.reshape(-1) - train_batch[Postprocessing.VALUE_TARGETS]) * mask, 2.0)) / batch_size
    overall_err = sum([
        0.1 * policy.pi_err,  # TODO tune lambda here
        # "vf_loss_coeff": 0.5  "entropy_coeff": 0.01,
        0.5 * policy.value_err,
        0.01 * policy.entropy,
        icm_loss  # TODO multiply by 10?
    ])
    return overall_err

# Fix estimation of final reward using value function to use internal recurrent
# state if any


def add_advantages(policy,
                   sample_batch,
                   other_agent_batches=None,
                   episode=None):
    _, exploration_rewards = policy.model.icm_forward(
        torch.Tensor(sample_batch[SampleBatch.OBS]),
        torch.Tensor(sample_batch[SampleBatch.NEXT_OBS]),
        torch.Tensor(sample_batch[SampleBatch.ACTIONS])
    )
    exploration_rewards = exploration_rewards.mean(dim=-1).numpy()
    sample_batch[SampleBatch.REWARDS] += exploration_rewards
    sample_batch['exploration_rewards'] = exploration_rewards

    completed = sample_batch[SampleBatch.DONES][-1]
    if completed:
        last_r = 0.0
    else:
        # Trajectory has been truncated, estimate final reward using the
        # value function from the terminal observation and
        # internal recurrent state if any
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append(sample_batch['state_out_{}'.format(i)][-1])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    return compute_advantages(
        sample_batch, last_r, policy.config['gamma'], policy.config['lambda'],
        policy.config['use_gae'], policy.config['use_critic'])

# Fix value network mixin to use internal recurrent state if any


class ValueNetworkMixin:
    def _value(self, obs, prev_action, prev_reward, *state):
        _ = self.model({
            SampleBatch.CUR_OBS: torch.Tensor([obs]).to(self.device),
            SampleBatch.PREV_ACTIONS: torch.Tensor([prev_action]).to(self.device),
            SampleBatch.PREV_REWARDS: torch.Tensor([prev_reward]).to(self.device),
        }, [torch.Tensor([s]).to(self.device) for s in state], torch.Tensor([1]).to(self.device))
        return self.model.value_function()[0]


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
        'vf_loss': policy.value_err.item(),
        'cur_lr': policy._optimizers[0].param_groups[0]['lr'],
        'icm_loss': policy.icm_loss.item(),
        'exploration_rewards': train_batch['exploration_rewards'].mean().item()
    }


def get_policy_class(config):
    return TunedA2CPolicy


TunedA2CPolicy = A3CTorchPolicy.with_updates(
    name='TunedA2CPolicy',
    get_default_config=lambda: TUNED_A2C_CONFIG,
    loss_fn=actor_critic_loss,
    stats_fn=stats,
    postprocess_fn=add_advantages,
    mixins=[ValueNetworkMixin],
    optimizer_fn=torch_optimizer)

TunedA2CTrainer = A2CTrainer.with_updates(
    name='TunedA2C',
    default_config=TUNED_A2C_CONFIG,
    default_policy=TunedA2CPolicy,
    get_policy_class=get_policy_class)
