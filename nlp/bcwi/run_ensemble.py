import os
import pickle
import collections

import torch
import numpy as np
from scipy.special import softmax

from experiment.utils import get_model_name_one
from model.evaluate import get_calculate_metrics_func
from model.utils import load_dataset, load_tokenizer
from experiment.utils import load_logits

seeds = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1010]

output_dir = os.path.join('..', 'outputs', 'v1')

dataset_names = ['MASSIVE', 'banking77', 'ag_news']  # MASSIVE, banking77, ag_news
scenarios = ['add_data', 'add_classes']  # add_data, add_classes
method = 'FT_old_on_updated_fullFT_bestACC'  # FT_old_on_updated_fullFT_bestACC
variant = 'plain'  # plain, multi

alphas = list(np.linspace(0.0, 0.3, 50))
alphas += list(np.linspace(0.3, 0.7, 300))
alphas += list(np.linspace(0.7, 1.0, 50))
#alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_new_models = [1, 2, 4, 8, 16]

pt_name = 'roberta-base'
load_tokenizer(pt_name)
method = method + '_ensemble' if variant == 'multi' else method

print(dataset_names, scenarios, method, variant)


def main():
    for dataset_name in dataset_names:
        for scenario in scenarios:
            for seed in seeds:
                print('seed', seed)
                dataset_dir = os.path.join('..', 'data', dataset_name)
                dataset_files = os.path.join(dataset_dir, scenario, 'updated', '{}.jsonl')
                dataset, dataset_info = load_dataset(['train', 'dev', 'test'], dataset_files)
                new_label_ids = dataset_info['new_label_ids'] if scenario == 'add_classes' else None

                old_model_dir = os.path.join(output_dir, dataset_name, str(seed), scenario, 'old_model')
                method_dir = os.path.join(output_dir, dataset_name, str(seed), scenario, method)

                if variant == 'plain':
                    run_plain(dataset, old_model_dir, method_dir, new_label_ids)

                if variant == 'multi':
                    run_multi(dataset, old_model_dir, method_dir, new_label_ids)

                if variant == 'logits':
                    run_plain(dataset, old_model_dir, method_dir, new_label_ids, logits=True)


def run_plain(dataset, old_model_dir, new_method_dir, new_label_ids):
    new_model_name = get_model_name_one(new_method_dir)
    metrics_func = get_calculate_metrics_func(old_model_dir)

    name = 'ensemble'
    metrics = average_probabilities(old_model_dir,
                                    new_method_dir,
                                    [new_model_name],
                                    alphas,
                                    dataset,
                                    metrics_func,
                                    new_label_ids=new_label_ids,
                                    )

    new_model_dir = os.path.join(new_method_dir, new_model_name)
    with open(os.path.join(new_model_dir, f'{name}.pickle'), 'wb') as f:
        pickle.dump(metrics, f)

    return metrics


def run_multi(dataset, old_model_dir, new_method_dir, new_label_ids):
    new_model_names = list()
    for dir_name in sorted(os.listdir(new_method_dir)):
        if dir_name.startswith('model'):
            new_model_names.append(dir_name)

    metrics_func = get_calculate_metrics_func(old_model_dir)

    all_metrics = dict()
    for num_models in num_new_models:
        print('num_models', num_models)

        metrics = average_probabilities(old_model_dir,
                                        new_method_dir,
                                        new_model_names[:num_models],
                                        alphas,
                                        dataset,
                                        metrics_func,
                                        new_label_ids=new_label_ids
                                        )
        all_metrics[num_models] = metrics


    all_metrics['num_new_models'] = num_new_models
    with open(os.path.join(new_method_dir, 'ensemble_multi.pickle'), 'wb') as f:
        pickle.dump(all_metrics, f)

    return all_metrics


def average_probabilities(old_model_dir, new_method_dir, new_model_names, alphas, dataset, metrics_func, new_label_ids=None):
    old_logits = load_logits(old_model_dir)
    assert len(old_logits) == 1
    old_logits = list(old_logits.values())[0]

    all_new_logits = list()
    for new_model_name in new_model_names:
        new_logits = load_logits(new_method_dir)
        new_logits = new_logits[new_model_name[6:]]
        all_new_logits.append(new_logits)

    alpha_metrics = collections.defaultdict(dict)
    for split in ['dev', 'test']:
        _dataset = dataset[split]
        dataset_labels = torch.as_tensor([e['label'] for e in _dataset])
        dataset_instance_ids = [e['id'] for e in _dataset]

        instance_ids, _old_logits, _ = old_logits[split]
        if new_label_ids:
            _old_logits[:, new_label_ids] = -np.inf
        assert instance_ids == dataset_instance_ids

        _all_new_logits = list()
        for new_logits in all_new_logits:
            instance_ids, _new_logits, _ = new_logits[split]
            assert instance_ids == dataset_instance_ids
            _all_new_logits.append(_new_logits)

        old_predictions = softmax(_old_logits, axis=-1)
        new_predictions = np.mean([softmax(_new_logits, axis=-1) for _new_logits in _all_new_logits], axis=0)

        for alpha in alphas:
            print('alpha', alpha)
            predictions = alpha * old_predictions + (1 - alpha) * new_predictions
            predictions = np.argmax(predictions, axis=-1)
            alpha_metrics[alpha][split] = metrics_func(split,
                                                       torch.from_numpy(predictions),
                                                       dataset_labels,
                                                       dataset_instance_ids,
                                                       strict=True)
            print(split, alpha_metrics[alpha][split])

    return alpha_metrics


if __name__ == '__main__':
    main()
