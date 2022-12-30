import os
import argparse

from model.utils import load_tokenizer, load_old_model_info
from experiment.utils import save_metrics, save_logits
from utils import setup_seed
from experiment.experiment_groups import exp_groups
from methods.main import get_run
from experiment.experiment_groups import hp

fp16 = True
parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--base_path', default='./', type=str, help='Base path of code.')
parser.add_argument('--exp_group', default=None, type=str, help='Name of the experiment group')
parser.add_argument('--datasets', nargs="+", default=['MASSIVE'], help='Name of the dataset; MASSIVE, banking77, ag_news')
parser.add_argument('--scenarios', nargs="+", default=['add_data'], help='Name of the data update scenario. add_data, add_classes')
parser.add_argument('--seeds', nargs="+", default=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1010],
                    help='Run experiments for different random seeds')
parser.add_argument('--output_dir', default='v1', type=str, help='output dir')
opts = parser.parse_args()

dataset_names = opts.datasets
scenarios = opts.scenarios
output_dir = os.path.join(opts.base_path, 'outputs', opts.output_dir)
seeds = [int(seed) for seed in opts.seeds]

if opts.exp_group is None:
    data_update_methods = [
        get_run(hp, from_model='old'),
    ]
else:
    data_update_methods = exp_groups[opts.exp_group]


def main():
    for scenario in scenarios:
        for dataset_name in dataset_names:
            dataset_dir = os.path.join(opts.base_path, 'data', dataset_name)
            dataset_files = os.path.join(dataset_dir, scenario, 'updated', '{}.jsonl')

            for seed in seeds:
                setup_seed(seed)

                old_model_dir = os.path.join(output_dir, dataset_name, str(seed), scenario, 'old_model')
                old_model_info = load_old_model_info(old_model_dir)
                load_tokenizer(old_model_info['pt_model_name'])

                for run_func, data_update_method_name in data_update_methods:

                    print('seed', seed)
                    print('dataset_name', dataset_name)
                    print('scenario', scenario)
                    print('pretrained_model_name', old_model_info['pt_model_name'])
                    print('data_update_method', data_update_method_name)
                    method_dir = os.path.join(output_dir, dataset_name, str(seed), scenario, data_update_method_name)
                    os.makedirs(method_dir, exist_ok=True)

                    metrics, logits = run_func(scenario, seed, dataset_files, old_model_dir, method_dir, fp16=fp16)
                    print('')
                    print(metrics)
                    if metrics:
                        save_metrics(method_dir, metrics)
                    if logits:
                        save_logits(method_dir, logits)

    print('finished experiments')


if __name__ == '__main__':
    main()
