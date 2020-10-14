import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

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
        self.fc1 = nn.Linear(3 * 3 * 32, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
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

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict['obs'].float().permute(0, 3, 1, 2)
        x = self.convnet(x)
        self.conv_features = x

        action_out = self.action_branch(self.conv_features)
        return action_out, state

    @override(TorchModelV2)
    def value_function(self):
        assert self.conv_features is not None, "must call forward() first"
        value_out = self.value_branch(self.conv_features).squeeze(1)
        return value_out
