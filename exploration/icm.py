"""
Adapted from https://github.com/rpatrik96/AttA2C
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ConvBlock(nn.Module):

    def __init__(self, ch_in=4):
        super().__init__()

        self.num_filter = 32
        self.size = 3
        self.stride = 2
        self.pad = self.size // 2

        # layers
        self.conv1 = nn.Conv2d(ch_in, self.num_filter, self.size, self.stride, self.pad)
        self.conv2 = nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad)
        self.conv3 = nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad)
        self.conv4 = nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        return x.reshape(x.shape[0], -1)  # retain batch size


class FeatureEncoderNet(nn.Module):
    def __init__(self, n_stack, in_size):
        
        super().__init__()
        # constants
        self.in_size = in_size
        self.h1 = 288

        # layers
        self.conv = ConvBlock(ch_in=n_stack)

    def forward(self, x):
        """
        In: [s_t]
            Current state (i.e. pixels) -> 1 channel image is needed
        Out: phi(s_t)
            Current state transformed into feature space
        :param x: input data representing the current state
        :return:
        """
        x = self.conv(x)
        return x.view(-1, self.in_size)


class ForwardNet(nn.Module):

    def __init__(self, in_size):
        """
        Network for the forward dynamics
        :param in_size: size(feature_space) + size(action_space)
        """
        super().__init__()

        # constants
        self.in_size = in_size
        self.fc_hidden = 256
        self.out_size = 288

        # layers
        self.fc1 = nn.Linear(self.in_size, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, self.out_size)

    def forward(self, x):
        """
        In: torch.cat((phi(s_t), a_t), 1)
            Current state transformed into the feature space,
            denoted by phi() and current action
        Out: \hat{phi(s_{t+1})}
            Predicted next state (in feature space)
        :param x: input data containing the concatenated current state in feature space
                  and the current action, pass torch.cat((phi(s_t), a_t), 1)
        :return:
        """
        return self.fc2(self.fc1(x))


class InverseNet(nn.Module):
    def __init__(self, num_actions, feat_size=288):
        """
        Network for the inverse dynamics
        :param num_actions: number of actions, pass env.action_space.n
        :param feat_size: dimensionality of the feature space (scalar)
        """
        super().__init__()

        # constants
        self.feat_size = feat_size
        self.fc_hidden = 256
        self.num_actions = num_actions

        # layers
        self.fc1 = nn.Linear(self.feat_size * 2, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, self.num_actions)

    def forward(self, x):
        """
        In: torch.cat((phi(s_t), phi(s_{t+1}), 1)
            Current and next states transformed into the feature space,
            denoted by phi().
        Out: \hat{a}_t
            Predicted action
        :param x: input data containing the concatenated current and next states, pass
                  torch.cat((phi(s_t), phi(s_{t+1}), 1)
        :return:
        """
        return self.fc2(self.fc1(x))

class ICMNet(nn.Module):
    def __init__(self, n_stack, num_actions, in_size=288, feat_size=256):
        """
        Network implementing the Intrinsic Curiosity Module (ICM) of https://arxiv.org/abs/1705.05363
        :param n_stack: number of frames stacked
        :param num_actions: dimensionality of the action space, pass env.action_space.n
        :param attn_target:
        :param attn_type:
        :param in_size: input size of the AdversarialHeads
        :param feat_size: size of the feature space
        """
        super().__init__()
        self.in_size = in_size  # pixels i.e. state
        self.feat_size = feat_size
        self.num_actions = num_actions

        self.feat_enc_net = FeatureEncoderNet(n_stack, self.in_size)
        self.fwd_net = ForwardNet(self.in_size + self.num_actions)
        self.inv_net = InverseNet(self.num_actions, self.in_size)

    def forward(self, current_states, next_states, action):
        """
        feature: current encoded state
        next_feature: next encoded state
        :param num_envs: number of the environments
        :param states: tensor of the states
        :param action: current action
        :return:
        icm_loss, intrinsic_reward
        """

        """Encode the states"""
        current_feature = self.feat_enc_net(current_states)
        next_feature = self.feat_enc_net(next_states)

        action_one_hot = torch.zeros(action.shape[0], self.num_actions, device=self.fwd_net.fc1.weight.device) \
            .scatter_(1, action.long().view(-1, 1), 1)
        fwd_in = torch.cat((current_feature, action_one_hot), 1)
        next_feature_pred = self.fwd_net(fwd_in)

        inv_in = torch.cat((current_feature, next_feature), 1)
        action_pred = self.inv_net(inv_in)

        return self._calc_loss(next_feature, next_feature_pred, action_pred, action), F.mse_loss(next_feature_pred, next_feature, reduction='none') * 0.5


    def _calc_loss(self, features, feature_preds, action_preds, actions):

        # forward loss
        loss_fwd = F.mse_loss(feature_preds, features)
        # inverse loss
        loss_inv = F.cross_entropy(action_preds.view(-1, self.num_actions), actions.long())

        return loss_fwd + loss_inv # TODO tune beta here