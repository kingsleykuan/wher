import math
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

class ManagerModule(nn.Module):
    """
    """
    def __init__(self, num_features_d, horizon):
        super(ManagerModule, self).__init__()

        self.num_features_d = num_features_d
        self.horizon = horizon

        self.fc_m_space = nn.Linear(num_features_d, num_features_d)
        self.m_rnn = nn.LSTM(num_features_d, num_features_d, batch_first=True)
        # Init forget gate bias to 1 for improved learning
        self.m_rnn.bias_ih_l0.data[num_features_d:num_features_d * 2].fill_(1)

        self.critic = nn.Linear(num_features_d, 1)

    def forward(self, z, state):
        # Input has shape [Batch, Time, Features]
        latent_state = F.relu(self.fc_m_space(z))

        horizon = self.horizon

        # Extract states and goals over past horizon
        h_horizon = state[:horizon]
        c_horizon = state[horizon:horizon * 2]
        goal_horizon = state[horizon * 2:]

        # Sample Batch
        if z.shape[1] == 1:
            # Dilated LSTM
            h = torch.unsqueeze(h_horizon[0], 0)
            c = torch.unsqueeze(c_horizon[0], 0)
            goal, [h, c] = self.m_rnn(latent_state, [h, c])

            # Update past horizon like a queue
            for i in range(horizon - 1):
                h_horizon[i] = h_horizon[i + 1]
                c_horizon[i] = c_horizon[i + 1]
                goal_horizon[i] = goal_horizon[i + 1]
            h_horizon[-1] = torch.squeeze(h, 0)
            c_horizon[-1] = torch.squeeze(c, 0)
            goal_horizon[-1] = goal

            # Pool goal over past horizon
            goal = torch.sum(torch.stack(goal_horizon), dim=0)
        # Train Batch
        else:
            goals = []
            # Dilated LSTM
            # 1) Splits training batch by taking every i-th element
            # 2) Takes initial state from past states horizon
            # 3) Runs LSTM over splits
            # 4) Interleaves outputs
            pad_len = float(latent_state.shape[1]) / float(horizon)
            pad_len = (math.ceil(pad_len) * horizon) - latent_state.shape[1]
            latent_state_padded = F.pad(latent_state, (0, 0, 0, pad_len))

            for i in range(horizon):
                latent_state_dilated = latent_state_padded[:,i::horizon,:]
                h = torch.unsqueeze(h_horizon[i], 0)
                c = torch.unsqueeze(c_horizon[i], 0)

                goal, [h, c] = self.m_rnn(latent_state_dilated, [h, c])
                goals.append(goal)

            goal = torch.stack(goals, dim=2).reshape(latent_state_padded.shape)
            goal = goal[:, :latent_state.shape[1], :]

            # Pool goal over past horizon
            # 1) Concat past goal horizon with goals
            # 2) Run 1D sum pooling to pool goals
            goal = torch.cat((torch.cat(goal_horizon[1:], dim=1), goal), dim=1)
            goal = goal.permute(0, 2, 1)
            goal = F.avg_pool1d(goal, horizon, 1) * horizon
            goal = goal.permute(0, 2, 1)

        goal = goal / goal.norm(dim=-1, keepdim=True)
        return latent_state, goal, h_horizon + c_horizon + goal_horizon

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

        hidden_size = num_features_k * num_actions
        self.w_rnn = nn.LSTM(num_features_d, hidden_size, batch_first=True)
        # Init forget gate bias to 1 for improved learning
        self.w_rnn.bias_ih_l0.data[hidden_size:hidden_size * 2].fill_(1)

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

class FuNModel(RecurrentNetwork, nn.Module):
    """
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, fun_horizon):
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.num_input_channels = 4
        self.num_features_d = 256
        self.num_features_k = 16
        self.num_outputs = num_outputs
        self.horizon = fun_horizon

        self.convnet = SmallConvNet(
            self.num_input_channels, self.num_features_d)
        self.conv_features = None

        self.z = None
        
        self.manager = ManagerModule(self.num_features_d, self.horizon)
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
        horizon = self.horizon

        worker_states = state[:2]
        manager_states = state[2:2 + horizon * 3]
        horizon_goals = state[2 + horizon * 3:]

        self.z = inputs

        latent_state, goal, manager_states = self.manager(
            inputs, manager_states)
        self.latent_state = latent_state
        self.goal = goal

        # Sample Batch
        if inputs.shape[1] == 1:
            # Update horizon like a queue
            for i in range(horizon - 1):
                horizon_goals[i] = horizon_goals[i + 1]
            horizon_goals[-1] = goal.detach()

            # Pool goal over past horizon
            horizon_goal = torch.sum(torch.stack(horizon_goals), dim=0)
        # Train Batch
        else:
            # Pool goal over past horizon
            # 1) Concat past goal horizon with goals
            # 2) Run 1D sum pooling to pool goals
            horizon_goal = torch.cat(
                (torch.cat(horizon_goals[1:], dim=1), goal.detach()), dim=1)
            horizon_goal = horizon_goal.permute(0, 2, 1)
            horizon_goal = F.avg_pool1d(horizon_goal, horizon, 1) * horizon
            horizon_goal = horizon_goal.permute(0, 2, 1)

        action, worker_states = self.worker(inputs, horizon_goal, worker_states)
        return action, worker_states + manager_states + horizon_goals

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
                1, self.num_features_k * self.num_outputs).zero_().squeeze(0),
            self.convnet.fc1.weight.new(
                1, self.num_features_k * self.num_outputs).zero_().squeeze(0),
        ]

        horizon = self.horizon

        # Manager Dilated LSTM States
        for _ in range(horizon * 2):
            h.append(self.convnet.fc1.weight.new(
                1, self.num_features_d).zero_().squeeze(0))

        # Manager Goals Horizon
        for _ in range(horizon):
            h.append(self.convnet.fc1.weight.new(
                1, 1, self.num_features_d).zero_().squeeze(0))

        # Horizon Goals
        for _ in range(horizon):
            h.append(self.convnet.fc1.weight.new(
                1, 1, self.num_features_d).zero_().squeeze(0))

        return h

    def manager_features(self):
        assert self.latent_state is not None, "must call forward() first"
        assert self.goal is not None, "must call forward() first"
        return self.latent_state, self.goal
