import os
import pickle
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification
from torch.utils.data import DataLoader


from model.utils import load_dataset, custom_collate_fn, device
from model.utils import load_old_model_info, load_tokenizer
import time


def get_diag_fisher(old_model, dataset, old_instance_ids, splits=['train'], scale=False, full=False, old_only=False, new_label_ids=None):
    """
    Calculate the diagonal of the empirical Fisher information matrix for the old model.
    """

    diag_fisher = dict()
    for name, p in old_model.named_parameters():
        diag_fisher[name] = torch.zeros_like(p)

    num_processed_instances = 0
    print('generate diag fisher')
    for split in splits:
        # Pass single instances through the model to get isolated gradient for this instance
        dataloader = DataLoader(dataset[split], collate_fn=custom_collate_fn, batch_size=1)
        for batch in dataloader:
            assert len(batch['instance_ids']) == 1
            # skip instances with new classes. Old model was not trained on them.
            if old_only and batch['instance_ids'][0] not in old_instance_ids[split]:
                continue

            old_model.zero_grad()

            model_inputs = dict(input_ids=batch['input_ids'].to(device),
                                attention_mask=batch['attention_mask'].to(device))
            outputs = old_model(**model_inputs)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Iterate over all possible labels
            if full:
                for label_id in range(logits.shape[1]):
                    if new_label_ids and label_id in new_label_ids:
                        continue
                    prob = probs[0, label_id].item()
                    label = torch.tensor([label_id]).to(device)
                    loss = F.nll_loss(log_probs, label)
                    # retain_graph allows to run multiple backward passes
                    loss.backward(retain_graph=True)

                    for name, p in old_model.named_parameters():
                        diag_fisher[name] += (torch.square(p.grad.data) * prob)
                if num_processed_instances % 100 == 0:
                    print(num_processed_instances)
            # only use the gradient of the predicted label
            else:
                label = torch.argmax(logits, dim=-1)
                loss = F.nll_loss(log_probs, label)
                loss.backward()

                for name, p in old_model.named_parameters():
                    # squared gradient
                    diag_fisher[name] += torch.square(p.grad.data)

            num_processed_instances += 1
    print(num_processed_instances)

    # Scale to have mean of 1
    if scale:
        mean = torch.mean(torch.cat([v.flatten() for v in diag_fisher.values()]))
        for name in list(diag_fisher.keys()):
            diag_fisher[name] = diag_fisher[name] / mean
    # don't scale
    else:
        for name in list(diag_fisher.keys()):
            diag_fisher[name] = diag_fisher[name] / float(num_processed_instances)

    return diag_fisher


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the Fisher information matrix and store it in the directory of the old model')
    parser.add_argument('--base_path', default='./', type=str, help='Base path of code.')
    parser.add_argument('--datasets', nargs="+", default=['MASSIVE'], help='Names of the datasets; MASSIVE, banking77, ag_news')
    parser.add_argument('--scenarios', nargs="+", default=['add_data'], help='Names of the data update scenarios. add_data, add_classes')
    parser.add_argument('--seeds', nargs="+", default=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1010], help='Run for different random seeds')
    parser.add_argument('--output_dir', default='v1', type=str, help='output dir')
    opts = parser.parse_args()

    dataset_names = opts.datasets
    scenarios = opts.scenarios
    output_dir = os.path.join(opts.base_path, 'outputs', opts.output_dir)
    seeds = [int(seed) for seed in opts.seeds]

    pt_model_name = 'roberta-base'
    load_tokenizer(pt_model_name)

    for dataset_name in dataset_names:
        for scenario in scenarios:
            for seed in seeds:
                print('seed', seed)
                t = time.time()
                dataset_dir = os.path.join(opts.base_path, 'data', dataset_name)
                dataset_files = os.path.join(dataset_dir, scenario, 'updated', '{}.jsonl')
                dataset, dataset_info = load_dataset(['train', 'dev', 'test'], dataset_files)
                new_label_ids = dataset_info['new_label_ids'] if scenario == 'add_classes' else None

                old_model_dir = os.path.join(output_dir, dataset_name, str(seed), scenario, 'old_model')

                old_model = RobertaForSequenceClassification.from_pretrained(os.path.join(old_model_dir, 'model'))
                old_model.to(device)

                old_model_info = load_old_model_info(old_model_dir)
                old_instance_ids = old_model_info['old_instance_ids']
                old_diag_fisher = get_diag_fisher(old_model,
                                                  dataset,
                                                  old_instance_ids,
                                                  full=False,
                                                  splits=['train', 'dev'],
                                                  scale=True,
                                                  new_label_ids=new_label_ids)

                name = 'fisher.pickle'
                out_file = os.path.join(old_model_dir, name)
                with open(out_file, 'wb') as f:
                    pickle.dump(old_diag_fisher, f)
                print('wrote fisher to', out_file)
                print('time', time.time() - t)