import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from exploration.icm import ICMNet

class SmallConvNet(nn.Module):
    """
    Small PyTorch CNN.
    """
    def __init__(self):
        super(SmallConvNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv5 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv6 = nn.Conv2d(32, 32, 3, 2, 1)
        self.fc1 = nn.Linear(3 * 3 * 32, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.reshape((-1, 3 * 3 * 32))
        x = F.relu(self.fc1(x))
        return x

class SmallConvModel(TorchModelV2, nn.Module):
    """
    RLlib small CNN actor critic model.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.convnet = SmallConvNet()
        self.conv_features = None
        self.action_branch = nn.Linear(256, num_outputs)
        self.value_branch = nn.Linear(256, 1)
        
        self.icm_net = ICMNet(4, num_outputs, in_size=288, feat_size=256)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict['obs'].float().permute(0, 3, 1, 2) # 16 threads * 16 envs * 20 states -> 256 * 19 * 2
        x = self.convnet(x)
        self.conv_features = x

        action_out = self.action_branch(self.conv_features)
        return action_out, state


    @override(TorchModelV2)
    def value_function(self):
        assert self.conv_features is not None, "must call forward() first"
        value_out = self.value_branch(self.conv_features).squeeze(1)
        return value_out

    def icm_forward(self, obs, next_obs):
        return self.icm_net(obs.permute(0, 3, 1, 2).float(), next_obs.permute(0, 3, 1, 2).float())

    def icm_fwd_forward(self, actions):
        return self.icm_net.fwd_forward(actions)

    def icm_inv_forward(self, actions):
        return self.icm_net.inv_forward(actions)
