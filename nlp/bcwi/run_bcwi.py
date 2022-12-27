import os
import pickle
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from transformers import RobertaForSequenceClassification

from experiment.utils import get_model_dir_one
from model.evaluate import get_calculate_metrics_func
from model.utils import load_dataset, get_config, device, load_tokenizer
from bcwi.utils import interpolate_weights


parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--base_path', default='./', type=str, help='Base path of code.')
parser.add_argument('--variant', default='plain', choices=['plain', 'fisher', 'multi'], help='Name of the variant; plain, fisher, multi')
parser.add_argument('--datasets', nargs="+", default=['MASSIVE'], help='Name of the dataset; MASSIVE, banking77, ag_news')
parser.add_argument('--scenarios', nargs="+", default=['add_data'], help='Name of the data update scenario. add_data, add_classes')
parser.add_argument('--seeds', nargs="+", default=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1010],
                    help='Run experiments for different random seeds')
parser.add_argument('--output_dir', default='v1', type=str, help='output dir')
opts = parser.parse_args()

dataset_names = opts.datasets
scenarios = opts.scenarios
variant = opts.variant  # plain, multi, fisher
output_dir = os.path.join(opts.base_path, 'outputs', opts.output_dir)
seeds = [int(seed) for seed in opts.seeds]

method = 'FT_old_on_updated_fullFT_bestACC'  # FT_old_on_updated_fullFT_bestACC

if variant == 'fisher':
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
else:
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
                dataset_dir = os.path.join(opts.base_path, 'data', dataset_name)
                dataset_files = os.path.join(dataset_dir, scenario, 'updated', '{}.jsonl')
                dataset, dataset_info = load_dataset(['train', 'dev', 'test'], dataset_files)
                new_label_ids = dataset_info['new_label_ids'] if scenario == 'add_classes' else None

                old_model_dir = os.path.join(output_dir, dataset_name, str(seed), scenario, 'old_model')
                method_dir = os.path.join(output_dir, dataset_name, str(seed), scenario, method)

                if variant == 'plain':
                    run_plain(dataset, old_model_dir, method_dir, seed, new_label_ids)

                if variant == 'multi':
                    run_multi(dataset, old_model_dir, method_dir, seed, new_label_ids)

                if variant == 'fisher':
                    run_fisher(dataset, old_model_dir, method_dir, seed, new_label_ids)


def run_fisher(dataset, old_model_dir, method_dir, seed, new_label_ids):
    new_model_dir = get_model_dir_one(method_dir)
    metrics_func = get_calculate_metrics_func(old_model_dir)
    config = get_config(dict(), seed)

    old_model = RobertaForSequenceClassification.from_pretrained(os.path.join(old_model_dir, 'model'))
    old_model.to(device)

    new_model = RobertaForSequenceClassification.from_pretrained(new_model_dir)
    new_model.to(device)

    with open(os.path.join(old_model_dir, 'fisher.pickle'), 'rb') as f:
        old_diag_fisher = pickle.load(f)

    metrics = interpolate_weights(old_model,
                                  [new_model],
                                  alphas,
                                  dataset,
                                  config,
                                  metrics_func,
                                  new_label_ids=new_label_ids,
                                  weighted=old_diag_fisher,
                                  )

    out_file = os.path.join(new_model_dir, 'fisher.pickle')
    print('save to: ', out_file)
    print('')
    with open(out_file, 'wb') as f:
        pickle.dump(metrics, f)

    return metrics


def run_multi(dataset, old_model_dir, method_dir, seed, new_label_ids):

    metrics_func = get_calculate_metrics_func(old_model_dir)
    new_models = list()
    config = get_config(dict(), seed)

    for dir_name in sorted(os.listdir(method_dir)):
        if dir_name.startswith('model'):
            new_model_dir = os.path.join(method_dir, dir_name)
            print('load', new_model_dir)
            new_model = RobertaForSequenceClassification.from_pretrained(new_model_dir)
            new_model.to(device)
            new_models.append(new_model)
            if len(new_models) == max(num_new_models):
                break
    print(len(new_models))
    assert len(new_models) > 1

    old_model = RobertaForSequenceClassification.from_pretrained(os.path.join(old_model_dir, 'model'))
    old_model.to(device)

    all_metrics = dict()

    for num_models in num_new_models:
        print('num_models', num_models)

        metrics = interpolate_weights(old_model,
                                      new_models[:num_models],
                                      alphas,
                                      dataset,
                                      config,
                                      metrics_func,
                                      new_label_ids=new_label_ids)
        all_metrics[num_models] = metrics

    all_metrics['num_new_models'] = num_new_models
    with open(os.path.join(method_dir, 'bcwi_multi.pickle'), 'wb') as f:
        pickle.dump(all_metrics, f)

    return all_metrics


def run_plain(dataset, old_model_dir, method_dir, seed, new_label_ids):
    new_model_dir = get_model_dir_one(method_dir)
    metrics_func = get_calculate_metrics_func(old_model_dir)
    config = get_config(dict(), seed)

    old_model = RobertaForSequenceClassification.from_pretrained(os.path.join(old_model_dir, 'model'))
    new_model = RobertaForSequenceClassification.from_pretrained(new_model_dir)

    old_model.to(device)
    new_model.to(device)

    metrics = interpolate_weights(old_model,
                                  [new_model],
                                  alphas,
                                  dataset,
                                  config,
                                  metrics_func,
                                  new_label_ids=new_label_ids)

    out_file = os.path.join(new_model_dir, 'bcwi.pickle')
    print('save to: ', out_file)
    print('')
    with open(out_file, 'wb') as f:
        pickle.dump(metrics, f)

    return metrics


if __name__ == '__main__':
    main()
