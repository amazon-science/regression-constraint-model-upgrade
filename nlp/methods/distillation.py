import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

from model.utils import device, load_old_model_info
from experiment.utils import load_logits

# Distillation method based on https://arxiv.org/abs/2105.03048


def validate_argument(distill, name):
    if not distill:
        return distill, name

    if distill == 'better':
        distill = dict(variant='better')
    elif distill == 'all':
        distill = dict(variant='all')
    elif not isinstance(distill, dict):
        distill = dict()

    default_variant = 'better'
    default_alpha = 0.5
    default_temperature = 2.0
    distill['variant'] = distill.get('variant', default_variant)
    distill['alpha'] = distill.get('alpha', default_alpha)
    distill['temperature'] = distill.get('temperature', default_temperature)

    assert distill['variant'] in ['all', 'better', 'oldOnly', 'oldOnlyA']

    name += '_distill'
    name += distill['variant'].title()
    name += '_a' + str(distill['alpha'])
    name += '_t' + str(distill['temperature'])

    return distill, name


def apply(model, old_model_dir, distill, config, new_label_ids=None):
    if not distill:
        return model

    for key, value in distill.items():
        config[f'distill_{key}'] = value

    logits = load_logits(old_model_dir)
    assert len(logits) == 1
    logits = list(logits.values())[0]

    old_instance_ids, old_logits, _ = logits['train']
    print('old_instance_ids', len(old_instance_ids))
    old_logits = torch.from_numpy(old_logits).to(device)
    old_logits.requires_grad = False
    idx_pos = {idx: pos for pos, idx in enumerate(old_instance_ids)}

    old_model_info = load_old_model_info(old_model_dir)
    old_instance_ids = old_model_info['old_instance_ids']

    model = DistillationModel(
        model,
        old_logits,
        idx_pos,
        old_instance_ids=old_instance_ids,
        variant=config['distill_variant'],
        alpha=config['distill_alpha'],
        temperature=config['distill_temperature'],
        new_label_ids=new_label_ids
    )

    return model


class DistillationModel(nn.Module):
    """Distill the logits of the old model when training the new model."""

    def __init__(self, model, old_logits, idx_pos, old_instance_ids, variant='all', alpha=0.5, temperature=2.0, new_label_ids=None):
        super().__init__()
        self.model = model
        self.classifier = self.model.classifier
        self.num_labels = self.model.num_labels
        self.variant = variant
        self.temperature = temperature
        self.new_label_ids = new_label_ids
        self.alpha = alpha
        self.old_logits = old_logits
        self.idx_pos = idx_pos
        self.old_instance_ids = old_instance_ids
        self.loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.requires_instance_ids = True

    def forward(self, input_ids, attention_mask, labels, instance_ids, split, **kwargs):

        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        student_logits = outputs.logits
        student_loss = outputs.loss

        model_log = dict()

        if self.training:
            assert split == 'train'
            model_log['Distill_loss/student'] = student_loss.item()
            indices = torch.tensor([self.idx_pos[idx] for idx in instance_ids]).to(device)
            teacher_logits = torch.index_select(self.old_logits, dim=0, index=indices)

            assert student_logits.shape == teacher_logits.shape

            if self.variant == 'better':
                # Only distill when the old model makes better predictions than the new model
                teacher_loss = F.cross_entropy(teacher_logits, labels, reduction='none')
                student_loss = F.cross_entropy(student_logits, labels, reduction='none')
                index_student_better = (teacher_loss > student_loss).nonzero().squeeze(1)
                index_teacher_better = (~(teacher_loss > student_loss)).nonzero().squeeze(1)

                student_is_better = student_loss[index_student_better]
                student_is_worse = student_loss[index_teacher_better]
                _student_logits = student_logits[index_teacher_better]
                _teacher_logits = teacher_logits[index_teacher_better]

                loss_logit = self.loss_fct(F.log_softmax(_student_logits / self.temperature, dim=-1),
                                           F.softmax(_teacher_logits / self.temperature, dim=-1)) * (
                                         self.temperature ** 2)
                total_loss = (self.alpha * loss_logit + (1.0 - self.alpha) * student_is_worse.sum()) + student_is_better.sum()
                total_loss /= student_logits.shape[0]

            elif self.variant.startswith('oldOnly'):
                # Only distill old instances
                old_instances_indices = [i for i, i_id in enumerate(instance_ids) if i_id in self.old_instance_ids['train']]
                new_instances_indices = [i for i, i_id in enumerate(instance_ids) if i_id not in self.old_instance_ids['train']]
                old_student_logits = student_logits[old_instances_indices]
                old_teacher_logits = teacher_logits[old_instances_indices]

                new_student_logits = student_logits[new_instances_indices]
                new_labels = labels[new_instances_indices]

                new_student_loss = F.cross_entropy(new_student_logits, new_labels)
                old_loss_logit = self.loss_fct(F.log_softmax(old_student_logits / self.temperature, dim=-1),
                                           F.softmax(old_teacher_logits / self.temperature, dim=-1)) * (
                                         self.temperature ** 2)

                if self.variant == 'oldOnlyA':
                    # Only distill old instances and have a weighting parameter to trade off distill and model loss
                    new_teacher_logits = teacher_logits[new_instances_indices]
                    new_loss_logit = self.loss_fct(F.log_softmax(new_student_logits / self.temperature, dim=-1),
                                                   F.softmax(new_teacher_logits / self.temperature, dim=-1)) * (
                            self.temperature ** 2)
                    new_student_loss = self.alpha * new_loss_logit + (1.0 - self.alpha) * new_student_loss

                total_loss = 0
                if old_student_logits.shape[0] > 0:
                    total_loss += old_loss_logit
                if new_student_logits.shape[0] > 0:
                    total_loss += new_student_loss

            elif self.variant == 'all':
                # Distill all instances no matter what
                loss_logit = self.loss_fct(F.log_softmax(student_logits / self.temperature, dim=-1),
                                           F.softmax(teacher_logits / self.temperature, dim=-1)) * (self.temperature ** 2)
                total_loss = self.alpha * loss_logit + (1.0 - self.alpha) * student_loss
            else:
                raise ValueError()
        else:
            total_loss = student_loss

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=student_logits,
            hidden_states=None,
            attentions=model_log
        )

    def save_pretrained(self, output_dir):
        self.model.save_pretrained(output_dir)

    def state_dict(self):
        return self.model.state_dict()

