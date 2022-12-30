import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# IA^3 method from https://arxiv.org/abs/2205.05638
# IA^3 finetunes a model by introducing new trainable parameters that scale the activations after specific linear
# layers. The linear layers with scaling parameters are the key and value transformations in the attention mechanism,
# as well as after the non-linearity of the position-wise feed forward network. All the proper model weights are frozen.


def validate_argument(ia3, name):
    if not ia3:
        return ia3, name

    if ia3 == 'base':
        ia3 = dict(variant='base')
    elif not isinstance(ia3, dict):
        ia3 = dict()

    default_variant = 'base'
    ia3['variant'] = ia3.get('variant', default_variant)

    assert ia3['variant'] in ['base']

    name += '_ia3'
    name += ia3['variant'].title()

    return ia3, name


def apply(model, ia3, config, old_label_ids=None):
    """Modify model to include IA^3 linear layers"""

    if not ia3:
        return model

    for key, value in ia3.items():
        config[f'ia3_{key}'] = value

    # Regex to match the names of linear layers that should be replaced with IA^3 linear layer module
    lora_modules = ".*layer\.\d+\.attention\.self|.*layer\.\d+\.output"
    lora_layers = "key|value|dense"

    # Regex to match the name of weights that should be trainable
    trainable_param_names = ".*lora_b.*"

    # Iterate all components of the network and replace specific linear layers
    for m_name, module in dict(model.named_modules()).items():
        if re.fullmatch(lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    #print(m_name, c_name)
                    setattr(module, c_name, LoRALinear(layer))

    # Freeze all model weights except the IA^3 scaling parameters
    for (param_name, param) in model.named_parameters():
        if not re.fullmatch(trainable_param_names, param_name) and not 'gate' in param_name:
            if old_label_ids is not None and param_name.startswith('classifier.out_proj'):
                continue
            param.requires_grad = False
            #print('freeze', param_name)
        #else:
            #print('not freeze', param_name)


    model.classifier = IA3RobertaClassificationHead(model.classifier)

    return model


class LoRALinear(nn.Module):
    """Linear layer with learnable scaling parameters"""

    def __init__(self, linear_layer):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features))

    def forward(self, input):
        hidden = F.linear(input, self.weight, self.bias)
        hidden = hidden * self.multi_lora_b
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, rank={}, scaling_rank={}".format(
            self.in_features, self.out_features, self.bias is not None, self.rank, self.scaling_rank
        )


class IA3RobertaClassificationHead(nn.Module):
    """Modify Classification head to include scaling parameters after activations"""

    def __init__(self, classifier):
        super().__init__()
        self.dense = classifier.dense
        self.multi_lora_b = nn.Parameter(torch.ones(self.dense.out_features))
        self.multi_lora_b2 = nn.Parameter(torch.ones(self.dense.out_features))
        self.dropout = classifier.dropout
        self.out_proj = classifier.out_proj

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x * self.multi_lora_b
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = x * self.multi_lora_b2
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
