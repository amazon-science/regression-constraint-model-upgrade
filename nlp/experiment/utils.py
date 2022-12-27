import json
import os
import pickle
import random

import torch
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_metrics(method_dir, all_metrics):
    metrics_file = os.path.join(method_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print('saved metrics to: {}'.format(metrics_file))
    return all_metrics


def save_logits(method_dir, all_logits):
    logits_file = os.path.join(method_dir, 'logits.pickle')
    with open(logits_file, 'wb') as f:
        pickle.dump(all_logits, f)
    print('saved logits to: {}'.format(logits_file))
    return all_logits


def load_logits(method_dir):
    print(method_dir)
    logits_file = os.path.join(method_dir, 'logits.pickle')
    with open(logits_file, 'rb') as f:
        logits = pickle.load(f)
    return logits


def get_model_dir_one(method_dir):
    model_dirs = list()
    for dir in os.listdir(method_dir):
        if dir.startswith('model'):
            model_dirs.append(dir)
    assert len(model_dirs) == 1
    return os.path.join(method_dir, model_dirs[0])


def get_model_name_one(method_dir):
    model_dirs = list()
    for dir in os.listdir(method_dir):
        if dir.startswith('model'):
            model_dirs.append(dir)
    assert len(model_dirs) == 1
    return model_dirs[0]
