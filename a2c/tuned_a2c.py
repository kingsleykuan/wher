from torch import nn
from torch import optim
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch

# Merge modified config with A2C and A3C default config
TUNED_A2C_CONFIG = A2CTrainer.merge_trainer_configs(
    A2C_DEFAULT_CONFIG,
    {
        'use_gae': False,
        'lr': 7e-4,
        'grad_clip': 0.5,
        'epsilon': 1e-8,
    },
    _allow_unknown_configs=True,
)

# Modify losses to average over batches instead of sum
# Provides consistent behaviour when changing batch sizes
def actor_critic_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    values = model.value_function()
    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS])
    policy.entropy = dist.entropy().mean()
    policy.pi_err = -(train_batch[Postprocessing.ADVANTAGES] * log_probs.reshape(-1)).mean()
    policy.value_err = nn.functional.mse_loss(
        values.reshape(-1), train_batch[Postprocessing.VALUE_TARGETS])
    overall_err = sum([
        policy.pi_err,
        policy.config["vf_loss_coeff"] * policy.value_err,
        -policy.config["entropy_coeff"] * policy.entropy,
    ])
    return overall_err

# Use RMSprop as per source paper
# More consistent than ADAM in non-stationary problems such as RL
def torch_rmsprop_optimizer(policy, config):
    return optim.RMSprop(
        policy.model.parameters(),
        lr=config['lr'],
        eps=config['epsilon'])

# Update stats function to include the current learning rate
def stats(policy, train_batch):
    return {
        'policy_entropy': policy.entropy.item(),
        'policy_loss': policy.pi_err.item(),
        'vf_loss': policy.value_err.item(),
        'cur_lr': policy._optimizers[0].param_groups[0]['lr'],
    }

def get_policy_class(config):
    return TunedA2CPolicy

TunedA2CPolicy = A3CTorchPolicy.with_updates(
        name='TunedA2CPolicy',
        get_default_config=lambda: TUNED_A2C_CONFIG,
        loss_fn=actor_critic_loss,
        stats_fn=stats,
        optimizer_fn=torch_rmsprop_optimizer)

TunedA2CTrainer = A2CTrainer.with_updates(
    name='TunedA2C',
    default_config=TUNED_A2C_CONFIG,
    default_policy=TunedA2CPolicy,
    get_policy_class=get_policy_class)
