import os

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaForSequenceClassification

from model.utils import device


def validate_argument(l2, name):
    if not l2:
        return l2, name
    name += '_l2_' + str(l2)
    return l2, name


def apply(model, l2, config, old_model_dir, skip_classifier):
    if not l2:
        return model

    config['l2'] = l2

    old_model = RobertaForSequenceClassification.from_pretrained(os.path.join(old_model_dir, 'model'))
    old_model.to(device)
    model = L2RegularizedModel(model, old_model, l2, skip_classifier)

    return model


class L2RegularizedModel(nn.Module):

    def __init__(self, model, old_model, alpha, skip_classifier):
        super(L2RegularizedModel, self).__init__()
        self.model = model
        self.old_model = old_model
        self.old_state_dict = old_model.state_dict()
        self.classifier = self.model.classifier
        self.alpha = alpha
        self.skip_classifier = skip_classifier
        print('skip_classifier', skip_classifier)

    def forward(self, input_ids, attention_mask, labels):

        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        logits = outputs.logits
        loss = outputs.loss

        for name, p in self.model.named_parameters():
            if self.skip_classifier and name.startswith('classifier'):
                continue
            assert self.old_state_dict[name].requires_grad is False

            loss += self.alpha * torch.square(p - self.old_state_dict[name].data).sum()

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

    def save_pretrained(self, output_dir):
        self.model.save_pretrained(output_dir)

    def state_dict(self):
        return self.model.state_dict()
