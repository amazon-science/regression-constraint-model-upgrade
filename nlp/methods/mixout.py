import os
import types

from typing import Optional
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
from torch import nn
from transformers import RobertaForSequenceClassification

from model.utils import device

# Mixout (https://arxiv.org/abs/1909.11299) regularization for finetuning. Randomly replaces some of the current
# weights with the weights of the old model.


def validate_argument(mixout_prob, name):
    if mixout_prob is not None:
        name += '_mixout' + str(mixout_prob)
    return mixout_prob, name


def apply(model, mixout_prob, config, old_model_dir, skip_classifier=False):
    if mixout_prob is None or mixout_prob == 0:
        return model

    config['mixout'] = mixout_prob

    # need to keep old model weights in memory to construct weight replacement mask
    old_model = RobertaForSequenceClassification.from_pretrained(os.path.join(old_model_dir, 'model'))
    old_model.to(device)
    old_model.eval()
    old_state_dict = old_model.state_dict()

    for name, module in model.named_modules():
        if skip_classifier and name.startswith('classifier'):
            continue

        # Modify all layers in the model to include the Mixout step.
        if isinstance(module, nn.Dropout):
            module.p = 0
        elif isinstance(module, nn.Linear):
            old_weight = old_state_dict[name + '.weight'].detach()
            old_bias = old_state_dict[name + '.bias'].detach()
            _monkey_patch_linear(module, old_weight, old_bias, mixout_prob)
        elif isinstance(module, nn.LayerNorm):
            old_weight = old_state_dict[name + '.weight'].detach()
            old_bias = old_state_dict[name + '.bias'].detach()
            _monkey_patch_layerNorm(module, old_weight, old_bias, mixout_prob)
        elif isinstance(module, nn.Embedding):
            old_weight = old_state_dict[name + '.weight'].detach()
            _monkey_patch_embeddings(module, old_weight, mixout_prob)

    return model


def _monkey_patch_linear(linear, old_weight, old_bias, mixout_prob):
    # Monkey patch the forward function of the linear layer to include the mixout step.
    def forward(self, input):
        assert old_weight.requires_grad is False
        # randomly mixes current weights with the weights of the old model
        mixed_weight = _mixout(self.weight, old_weight, mixout_prob, self.training)
        mixed_bias = None
        if linear.bias is not None:
            mixed_bias = _mixout(self.bias, old_bias, mixout_prob, self.training)
        # returns the output of the linear layer with the mixed weights and biases
        return F.linear(input, mixed_weight, mixed_bias)

    # Monkey patching allows to replace the forward function of an instantiated class object.
    linear.forward = types.MethodType(forward, linear)


def _monkey_patch_layerNorm(layerNorm, old_weight, old_bias, mixout_prob):
    # Monkey patch the forward function of layer norm to include the mixout step.
    def forward(self, input):
        assert old_weight.requires_grad is False
        mixed_weight = _mixout(self.weight, old_weight, mixout_prob, self.training)
        mixed_bias = None
        if layerNorm.bias is not None:
            mixed_bias = _mixout(self.bias, old_bias, mixout_prob, self.training)
        return F.layer_norm(input, self.normalized_shape, mixed_weight, mixed_bias, self.eps)

    layerNorm.forward = types.MethodType(forward, layerNorm)


def _monkey_patch_embeddings(embeddings, old_weight, mixout_prob):
    # Monkey patch the forward function of the embedding layer to include the mixout step.
    def monkey(self, input):
        assert old_weight.requires_grad is False
        mixed_weight = _mixout(self.weight, old_weight, mixout_prob, self.training)
        return F.embedding(input,
                           mixed_weight,
                           self.padding_idx,
                           self.max_norm,
                           self.norm_type,
                           self.scale_grad_by_freq,
                           self.sparse)

    embeddings.forward = types.MethodType(monkey, embeddings)


def _mixout(input: torch.Tensor,
            target: Optional["OrderedDict[str, torch.Tensor]"] = None,
            p: float = 0.0,
            training: bool = False,
            inplace: bool = False) -> torch.Tensor:

    return Mixout.apply(input, target, p, training, inplace)


class Mixout(InplaceFunction):
    # based on https://github.com/bloodwass/mixout and
    # https://gist.github.com/stephenroller/f45a372e231825f9f5578e9e705f4e95
    # target: a weight tensor mixes with a input tensor
    # A forward method returns
    # [(1 - Bernoulli(1 - p) mask) * target + (Bernoulli(1 - p) mask) * input - p * target]/(1 - p)
    # where p is a mix probability of mixout.
    # A backward returns the gradient of the forward method.
    # Dropout is equivalent to the case of target=None.
    # I modified the code of dropout in PyTorch.
    @staticmethod
    def _make_noise(input: torch.Tensor) -> torch.Tensor:
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls,
                ctx,
                input: torch.Tensor,
                target: Optional["OrderedDict[str, torch.Tensor]"] = None,
                p: float = 0.0,
                training: bool = False,
                inplace: bool = False) -> torch.Tensor:

        if p < 0 or p > 1:
            raise ValueError(f"A mix probability of mixout has to be between 0 and 1,  but got {p}")

        if target is not None and input.size() != target.size():
            raise ValueError(
                f"A target tensor size must match with a input tensor size {input.size()}, but got {target.size()}")

        ctx.p = p
        ctx.training = training

        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p == 0 or not ctx.training:
            return output

        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], *([1] * (len(input.size()) - 1)))
        ctx.noise.expand_as(input)

        if ctx.p == 1:
            output = target.clone()
        else:
            output = ((1 - ctx.noise) * target + ctx.noise * output - ctx.p * target) / (1 - ctx.p)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Optional[torch.Tensor]:
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None
