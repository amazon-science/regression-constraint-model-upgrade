import torch
import math

import torch.nn as nn

from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init


class L2NormLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, temperature=0.01):
        super(L2NormLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature

        self.weight = Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        # normalize the input and output
        norm_input = input / torch.norm(input, dim=1, keepdim=True)
        norm_weights = self.weight / torch.norm(self.weight, dim=1, keepdim=True)

        return F.linear(norm_input, norm_weights) / self.temperature

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, temperature={}'.format(
            self.in_features, self.out_features, self.temperature
        )

