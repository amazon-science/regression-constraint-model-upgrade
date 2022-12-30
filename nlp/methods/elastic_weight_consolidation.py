import os
import pickle

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaForSequenceClassification

from model.utils import device

# Elastic Weight Consolidation (https://arxiv.org/pdf/1612.00796.pdf). Similar to L2 regularization but uses the
# diagonal of the fisher information matrix as importance score for the old model weights.


def validate_argument(ewc, name):
    if not ewc:
        return ewc, name
    name += '_ewc' + str(ewc)
    return ewc, name


def apply(model, ewc, config, old_model_dir, skip_classifier):
    if not ewc:
        return model

    config['ewc'] = ewc

    # Load the diagonal of the empirical fisher information matrix
    with open(os.path.join(old_model_dir, 'fisher.pickle'), 'rb') as f:
        old_diag_fisher = pickle.load(f)

    # Keep copy of old model weights in memory
    old_model = RobertaForSequenceClassification.from_pretrained(os.path.join(old_model_dir, 'model'),
                                                                 hidden_dropout_prob=0,
                                                                 attention_probs_dropout_prob=0).to(device)

    model = EWC(model, old_model, ewc, old_diag_fisher, skip_classifier)

    return model


class EWC(nn.Module):

    def __init__(self, model, old_model, alpha, diag_fisher, skip_classifier):
        super(EWC, self).__init__()
        self.model = model
        self.old_model = old_model
        self.old_state_dict = old_model.state_dict()
        self.classifier = self.model.classifier
        self.alpha = alpha
        self.diag_fisher = diag_fisher
        self.skip_classifier = skip_classifier
        print('skip_classifier', skip_classifier)

    def forward(self, input_ids, attention_mask, labels):

        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

        # EWC is only needed in training
        if not self.training:
            return outputs

        logits = outputs.logits
        loss = outputs.loss

        for name, p in self.model.named_parameters():
            if self.skip_classifier and name.startswith('classifier'):
                continue
            assert self.old_state_dict[name].requires_grad is False
            assert self.diag_fisher[name].requires_grad is False

            # L2 regularization weighted by FIM
            l2 = torch.square(p - self.old_state_dict[name].data)
            loss += self.alpha * (self.diag_fisher[name] * l2).sum()

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
