import ray
from ray import tune
from ray.rllib.models import ModelCatalog

from exploration.small_model import SmallConvModel
from exploration.tuned_a2c import TunedA2CTrainer

def main():
    config = {}

    config['framework'] = 'torch'
    config['num_gpus'] = 1

    # Effective batch size is 16*16*20 = 5120
    config['num_workers'] = 16
    config['num_envs_per_worker'] = 16
    config['rollout_fragment_length'] = 20

    config['env'] = tune.grid_search(
        [
            'BreakoutNoFrameskip-v4',
        ])
    config['min_iter_time_s'] = 0
    config['timesteps_per_iteration'] = 100000
    # config['monitor'] = True

    # Override policy config for experiments
    config['lr'] = 1e-3
    config['end_lr'] = 1e-4
    config['anneal_timesteps'] = 10000000
    config['grad_clip'] = 0.5
    # config['epsilon'] = tune.grid_search([1e-3, 1e-5, 1e-8])

    config['model'] = {}

    # Reduce input from 84x84 to 42x42 for faster training
    config['model']['dim'] = 42

    # Use custom model with more layers tuned for smaller input
    ModelCatalog.register_custom_model('small_conv_model', SmallConvModel)
    config['model']['custom_model'] = 'small_conv_model'

    # Use 1 main thread and 16 worker threads
    ray.init(num_cpus=17)
    tune.run(
        TunedA2CTrainer,
        config=config,
        stop={'timesteps_total': 10000000},
        checkpoint_freq=100,
        checkpoint_at_end=True)

if __name__ == "__main__":
    main()
