import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class PSoftPlus(nn.Module):
    def __init__(self, num_parameters=1):
        """
        Parametric softplus activation function
        :param num_parameters: Number of parameters
        """
        super(PSoftPlus, self).__init__()

        self.num_parameters = num_parameters
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(1))

    def forward(self, x):
        return self.weight * F.softplus(x)
