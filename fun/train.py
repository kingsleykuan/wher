import ray
from ray import tune
from ray.rllib.models import ModelCatalog

from fun.fun_model import FuNModel
from fun.fun_policy import FuNTrainer

def main():
    config = {}

    config['framework'] = 'torch'
    config['num_gpus'] = 1

    # Effective batch size is 16*16*20 = 5120
    config['num_workers'] = 16
    config['num_envs_per_worker'] = 8
    config['rollout_fragment_length'] = 50

    config['env'] = tune.grid_search(
        [
            'GravitarNoFrameskip-v4'
            # 'BreakoutNoFrameskip-v4',
            # 'MsPacmanNoFrameskip-v4',
            # 'MontezumaRevengeNoFrameskip-v4',
        ])
    config['min_iter_time_s'] = 0
    config['timesteps_per_iteration'] = 100000

    # config['evaluation_interval'] = 5
    # config['evaluation_num_episodes'] = 10
    # config['evaluation_config'] = { 'monitor': True }

    # Override policy config for experiments
    config['lr'] = 3e-4
    config['lr_mode'] = 'anneal'
    config['end_lr'] = 1e-4
    config['anneal_timesteps'] = 200000000
    config['grad_clip'] = 0.5
    # config['epsilon'] = tune.grid_search([1e-3, 1e-5, 1e-8])

    config['model'] = {}

    # Reduce input from 84x84 to 42x42 for faster training
    config['model']['dim'] = 42

    # Use custom model with more layers tuned for smaller input
    # Add lstm to model
    ModelCatalog.register_custom_model('FuNModel', FuNModel)
    config['model']['custom_model'] = 'FuNModel'
    config['model']['framestack'] = True
    config['model']['use_lstm'] = True
    config['model']['max_seq_len'] = 50

    # FuN Config
    config['fun_horizon'] = 10
    config['model']['custom_model_config'] = {}
    config['model']['custom_model_config']['fun_horizon'] = 10

    # Use 1 main thread and 16 worker threads
    ray.init(num_cpus=17)
    tune.run(
        FuNTrainer,
        config=config,
        stop={'timesteps_total': 200000000},
        checkpoint_freq=100,
        checkpoint_at_end=True)

if __name__ == "__main__":
    main()
