import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override

class SmallConvNet(nn.Module):
    """
    Small PyTorch CNN.
    """
    def __init__(self, num_input_channels, num_output_features):
        super(SmallConvNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_output_features = num_output_features

        self.conv1 = nn.Conv2d(num_input_channels, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 2, 1)
        self.fc1 = nn.Linear(3 * 3 * 32, num_output_features)

    def forward(self, x):
        # Input has shape [Batch, Channels, Height, Width]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape((-1, 3 * 3 * 32))
        x = F.relu(self.fc1(x))
        return x

# TODO: Dilated LSTM
class ManagerModule(nn.Module):
    """
    """
    def __init__(self, num_features_d):
        super(ManagerModule, self).__init__()

        self.num_features_d = num_features_d

        self.fc_m_space = nn.Linear(num_features_d, num_features_d)
        self.m_rnn = nn.LSTM(num_features_d, num_features_d, batch_first=True)

        self.critic = nn.Linear(num_features_d, 1)

    def forward(self, z, state):
        # Input has shape [Batch, Time, Features]
        latent_state = F.relu(self.fc_m_space(z))

        goal, [h, c] = self.m_rnn(latent_state,
            [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        goal = goal / goal.norm(dim=-1, keepdim=True)
        return latent_state, goal, [torch.squeeze(h, 0), torch.squeeze(c, 0)]
    
    def value_function(self, z):
        return self.critic(z)

class WorkerModule(nn.Module):
    """
    """
    def __init__(self, num_features_d, num_features_k, num_actions):
        super(WorkerModule, self).__init__()

        self.num_features_d = num_features_d
        self.num_features_k = num_features_k
        self.num_actions = num_actions

        self.phi = nn.Linear(num_features_d, num_features_k, bias=False)
        self.w_rnn = nn.LSTM(
            num_features_d, num_features_k * num_actions, batch_first=True)

        self.critic = nn.Linear(num_features_d, 1)

    def forward(self, z, goal, state):
        # Input has shape [Batch, Time, Features]
        goal_embedding = torch.unsqueeze(self.phi(goal), -1)

        embedding_matrix, [h, c] = self.w_rnn(z,
            [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        embedding_matrix = torch.reshape(
            embedding_matrix, [
                embedding_matrix.shape[0],
                embedding_matrix.shape[1],
                self.num_actions,
                self.num_features_k
            ])

        action = torch.squeeze(
            torch.matmul(embedding_matrix, goal_embedding), -1)
        
        return action, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    def value_function(self, z):
        return self.critic(z)

# TODO: Dilated LSTM states
# TODO: Pool g over horizon
class FuNModel(RecurrentNetwork, nn.Module):
    """
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.num_input_channels = 4
        self.num_features_d = 256
        self.num_features_k = 16
        self.num_outputs = num_outputs

        self.convnet = SmallConvNet(
            self.num_input_channels, self.num_features_d)
        self.conv_features = None

        self.z = None
        
        self.manager = ManagerModule(self.num_features_d)
        self.latent_state = None
        self.goal = None

        self.worker = WorkerModule(
            self.num_features_d, self.num_features_k, self.num_outputs)

    @override(RecurrentNetwork)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict['obs'].float().permute(0, 3, 1, 2)
        self.conv_features = self.convnet(x)

        input_dict["obs_flat"] = self.conv_features
        return super().forward(input_dict, state, seq_lens)

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        self.z = inputs

        latent_state, goal, [m_h, m_c] = self.manager(inputs, state[:2])
        self.latent_state = latent_state
        self.goal = goal

        action, [w_h, w_c] = self.worker(inputs, goal.detach(), state[2:])

        return action, [m_h, m_c, w_h, w_c]

    @override(ModelV2)
    def value_function(self):
        assert self.z is not None, "must call forward() first"
        manager_values = torch.reshape(self.manager.value_function(self.z), [-1])
        worker_values = torch.reshape(self.worker.value_function(self.z), [-1])
        return manager_values, worker_values

    @override(ModelV2)
    def get_initial_state(self):
        h = [
            self.convnet.fc1.weight.new(
                1, self.num_features_d).zero_().squeeze(0),
            self.convnet.fc1.weight.new(
                1, self.num_features_d).zero_().squeeze(0),
            self.convnet.fc1.weight.new(
                1, self.num_features_k * self.num_outputs).zero_().squeeze(0),
            self.convnet.fc1.weight.new(
                1, self.num_features_k * self.num_outputs).zero_().squeeze(0),
        ]
        return h
    
    def manager_features(self):
        assert self.latent_state is not None, "must call forward() first"
        assert self.goal is not None, "must call forward() first"
        return self.latent_state, self.goal
