import collections

import torch
from torch.utils.data import DataLoader
from datasets import load_metric
import numpy as np

from model.utils import custom_collate_fn, device, load_old_model_info
from experiment.utils import load_logits

accuracy_metric = load_metric("accuracy")


def evaluate(split, model, dataset, config, metrics_func=None, step=None, writer=None):
    model.eval()

    dataloader = DataLoader(dataset[split], collate_fn=custom_collate_fn, batch_size=config['eval_batch_size'])

    instance_ids = list()
    all_logits = list()
    all_labels = list()
    all_losses = list()
    all_model_logs = collections.defaultdict(list)

    for batch in dataloader:
        batch_size = len(batch['labels'])

        with torch.no_grad():
            model_inputs = dict(input_ids=batch['input_ids'].to(device),
                                attention_mask=batch['attention_mask'].to(device),
                                labels=batch['labels'].to(device))
            if hasattr(model, 'requires_instance_ids'):
                model_inputs['instance_ids'] = batch['instance_ids']
                model_inputs['split'] = split

            outputs = model(**model_inputs)

        instance_ids.extend(batch['instance_ids'])
        all_logits.append(outputs.logits)
        all_labels.append(batch['labels'])
        all_losses.append((outputs.loss.item(), batch_size))
        if outputs.attentions is not None:
            for key, value in outputs.attentions.items():
                all_model_logs[key].append(value)

    loss = sum(l * n for l, n in all_losses) / sum(n for l, n in all_losses)

    logits = torch.cat(all_logits)
    predictions = torch.argmax(logits, dim=-1)
    labels = torch.cat(all_labels)

    if metrics_func is not None:
        metrics = metrics_func(split, predictions, labels, instance_ids)
    else:
        metrics = calculate_metrics(split, predictions, labels, instance_ids)
    logits = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    if writer:
        writer.add_scalar(f'Loss/{split}', loss, global_step=step)
        for key, value in metrics.items():
            if value is None:
                continue
            writer.add_scalar(f'{key}/{split}', value, global_step=step)
        for key, values in all_model_logs.items():
            writer.add_scalar(key, np.mean(values), global_step=step)

    metrics['loss'] = loss

    return metrics, (instance_ids, logits, labels)


def get_calculate_metrics_func(old_model_dir):
    logits = load_logits(old_model_dir)
    assert len(logits) == 1
    logits = list(logits.values())[0]
    old_model_info = load_old_model_info(old_model_dir)

    old_instance_ids = old_model_info['old_instance_ids']
    new_instance_ids = old_model_info['new_instance_ids']

    old_model_outputs = dict()
    for split in ['train', 'dev', 'test']:
        instance_ids = logits[split][0]
        predictions = np.argmax(logits[split][1], axis=-1)
        labels = logits[split][2]

        assert len(instance_ids) == len(predictions)
        assert len(instance_ids) == len(labels)
        instance_id_to_idx = {instance_id: idx for idx, instance_id in enumerate(instance_ids)}

        old_model_outputs[split] = (instance_ids, instance_id_to_idx, predictions, labels)

    def _calculate_metrics(split, model_predictions, model_labels, model_instance_ids, strict=True):
        metrics = dict()
        _old_instance_ids = old_instance_ids[split]
        _new_instance_ids = new_instance_ids[split]

        model_labels = model_labels.cpu().numpy()
        model_predictions = model_predictions.cpu().numpy()

        model_outputs = model_predictions, model_labels, model_instance_ids
        if strict:  # do not assert when computing for a single batch in training
            assert len(model_instance_ids) == len(_old_instance_ids) + len(_new_instance_ids)
        acc_all = accuracy_metric.compute(predictions=model_predictions, references=model_labels)['accuracy'] * 100
        nfr_all, nfi_all, _acc_all = get_regression_metrics(old_model_outputs[split], model_outputs, strict)
        if _acc_all:
            assert np.isclose(acc_all, _acc_all)

        metrics['ACC'] = acc_all
        metrics['NFR'] = nfr_all
        metrics['NFI'] = nfi_all

        indices = [idx for idx, instance_id in enumerate(model_instance_ids) if instance_id in _old_instance_ids]
        if strict:  # do not assert when computing for a single batch in training
            assert len(indices) == len(_old_instance_ids)
        model_outputs = model_predictions[indices], model_labels[indices], [model_instance_ids[idx] for idx in indices]
        nfr_old, nfi_old, acc_old = get_regression_metrics(old_model_outputs[split], model_outputs, strict)

        metrics['ACC_OLD'] = acc_old
        metrics['NFR_OLD'] = nfr_old
        metrics['NFI_OLD'] = nfi_old

        indices = [idx for idx, instance_id in enumerate(model_instance_ids) if instance_id in _new_instance_ids]
        if strict:  # do not assert when computing for a single batch in training
            assert len(indices) == len(_new_instance_ids)
        model_outputs = model_predictions[indices], model_labels[indices], [model_instance_ids[idx] for idx in indices]
        nfr_new, nfi_new, acc_new = get_regression_metrics(old_model_outputs[split], model_outputs, strict)

        metrics['ACC_NEW'] = acc_new
        metrics['NFR_NEW'] = nfr_new
        metrics['NFI_NEW'] = nfi_new

        return metrics

    return _calculate_metrics


def get_regression_metrics(old_model_output, model_outputs, strict=True):
    """
    Calculates NFR and NFI regression metrics in respect to the old model.
    """
    all_instance_ids, all_instance_id_to_idx, old_model_predictions, labels = old_model_output
    model_predictions, model_labels, model_instance_ids = model_outputs

    # train batch can consist of only new instances
    # test can consist of only old instances
    if len(model_instance_ids) == 0:
        return None, None, None

    indices = [all_instance_id_to_idx[instance_id] for instance_id in model_instance_ids]
    old_model_predictions = old_model_predictions[indices]
    assert np.array_equal(model_labels, labels[indices])

    total = float(len(model_instance_ids))
    error_new_model = np.not_equal(model_predictions, model_labels)
    nf = np.logical_and(np.equal(old_model_predictions, model_labels), error_new_model).sum()
    error_new_model = error_new_model.sum()

    error_rate_new_model = error_new_model / total
    nfr = nf / total
    if error_new_model == 0:  # if the model is perfect then there is no regression
        nfi = 0.0
    else:
        nfi = nfr / error_rate_new_model
    acc = 1 - error_rate_new_model
    return nfr*100, nfi*100, acc*100


def calculate_metrics(split, predictions, labels, instance_ids=None, strict=None):
    if len(predictions) == 0:
        return dict(ACC=None)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']

    metrics = dict()
    metrics['ACC'] = acc * 100
    return metrics


def get_metrics(splits, model, dataset, config, metrics_func=None, silent=False):
    all_metrics = dict()
    all_logits = dict()
    for split in splits:
        if not silent:
            print('run eval for {} data'.format(split))
        metrics, (instance_ids, logits, labels) = evaluate(split, model, dataset, config, metrics_func)
        all_metrics[split] = metrics
        all_logits[split] = (instance_ids, logits, labels)
    return all_metrics, all_logits


def get_distance(old_model, new_model):
    # L2 distance between old and new model
    old_model.to(device)
    new_model.to(device)
    d = 0
    model1_state_dict = old_model.state_dict()
    model2_state_dict = new_model.state_dict()
    for key in model1_state_dict:
        if not (key.endswith('bias') or key.endswith('weight')):
            continue
        d += (model1_state_dict[key] - model2_state_dict[key]).pow(2).sum()
    d = d.sqrt()
    return d.detach().item()
