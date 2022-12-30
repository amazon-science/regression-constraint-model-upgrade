import os
import json
import pickle

import torch
import datasets

from transformers import RobertaTokenizer
#from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = []


def tokenize_function(examples):
    return tokenizer[0](examples['text'], padding=False, truncation=True, return_attention_mask=False)


def load_tokenizer(pt_model_name):
    if len(tokenizer) == 0:
        tokenizer.append(RobertaTokenizer.from_pretrained(pt_model_name))


def get_config(hparam, seed):
    config = dict()

    config['num_epochs'] = 10
    config['batch_size'] = 16
    config['eval_batch_size'] = 256
    config['warmup_ratio'] = 0.1
    config['learning_rate'] = 3e-5
    config['weight_decay'] = 0.01
    config['prior_wd'] = 0.0
    config['adam_beta1'] = 0.9
    config['adam_beta2'] = 0.98
    config['adam_epsilon'] = 1e-6
    config['max_grad_norm'] = 5.0
    config['seed'] = seed

    hparam_name_map = dict(e='num_epochs', lr='learning_rate', wd='weight_decay', bs='batch_size')
    for key, value in hparam.items():
        key = hparam_name_map.get(key, key)
        if key == 'seed':
            config[key] += value
        else:
            config[key] = value

    return config


"""
def get_writer(model_output_dir, gating=False, distill=False):
    writer = SummaryWriter(model_output_dir, max_queue=1000)
    layout = {'Metric': {'ACC': ['Multiline', ['ACC/train', 'ACC/dev']],
                         'NFR': ['Multiline', ['NFR/train', 'NFR/dev']],
                         'NFI': ['Multiline', ['NFI/train', 'NFI/dev']],
                         'Loss': ['Multiline', ['Loss/train', 'Loss/dev']]
                         }
              }
    if gating:
        layout['Gate'] = {'Alpha': ['Multiline', ['Gating_Alpha/train', 'Gating_Alpha/dev']]}
    if distill:
        layout['Distill'] = {'Loss': ['Multiline', ['Distill_loss/student', 'Distill_loss/teacher']]}
    writer.add_custom_scalars(layout)
    return writer
"""


def custom_collate_fn(features):
    batch = tokenizer[0].pad(features, padding='longest')

    batch['input_ids'] = torch.tensor(batch['input_ids'])
    batch['attention_mask'] = torch.tensor(batch['attention_mask'])
    batch['instance_ids'] = batch['id']
    del batch['id']
    batch['labels'] = torch.tensor(batch['label'], dtype=torch.long)
    del batch['label']

    return batch


def save_model_to_dir(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print('model save to: {}'.format(output_dir))


def save_hparams_to_dir(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    hparams_file = os.path.join(output_dir, 'hparams.json')
    with open(hparams_file, 'w') as f:
        json.dump(config, f, indent=2)
    print('saved hparams to: {}'.format(hparams_file))


def load_dataset(splits, file_template):
    data_files = dict()
    for split in splits:
        data_files[split] = file_template.format(split)

    dataset = datasets.load_dataset('json', data_files=data_files)
    with open(file_template.format('info')[:-1]) as f:
        dataset_info = json.load(f)

    new_label_ids = [dataset_info['labels'].index(c) for c in dataset_info['add_classes']]
    old_label_ids = [i for i, c in enumerate(dataset_info['labels']) if c not in dataset_info['add_classes']]
    dataset_info['new_label_ids'] = new_label_ids
    dataset_info['old_label_ids'] = old_label_ids

    tokenized_dataset = dataset.map(tokenize_function, batched=False)

    return tokenized_dataset, dataset_info


def load_old_model_info(old_model_dir):
    with open(os.path.join(old_model_dir, 'infos.pickle'), 'rb') as f:
        old_model_info = pickle.load(f)
    return old_model_info
