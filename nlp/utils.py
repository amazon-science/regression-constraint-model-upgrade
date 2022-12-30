import torch
import numpy as np
import random
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def upload_directory_to_s3(s3, bucket_name, prefix, local_dir):
    for path, subdirs, files in os.walk(local_dir):
        for file in files:
            full_path = os.path.join(path, file)
            with open(full_path, 'rb') as data:
                s3.upload_fileobj(data, bucket_name, prefix + full_path)