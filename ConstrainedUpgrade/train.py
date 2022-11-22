import networks as models
import networks.TYY_stodepth_lineardecay as stodepth_models
import argparse
import os
import io
import sys
import random
import shutil
import time
import warnings
import collections
import datetime
import boto3
import numpy as np

from datasets import CombineDataset, ModelOutputDataset, NFLikelihoodDataset, ImageListDataset
from losses import FocalDistillationLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sagemaker_training import environment

from torch_patch import all_gather as all_gather_with_ag
from transforms import (MultiCropToTensor, MultiCropNormalize)

sys.stdout.flush()

def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)


model_names = sorted([
    name for name in models.__dict__
    if not name.startswith("__") and callable(models.__dict__[name])
] + ['swintrans'])


class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '-d',
    '--data',
    default='~/resource/imagenet/',
    metavar='DIR',
    help='path to dataset')
parser.add_argument(
    '--train_img_txt',
    default=None,
    type=str,
    metavar='PATH',
    help='train images txt, samples should be listed as'
         'root/dog/xxx.png 0'
         '...'
         'root/cat/yyy.jpg 1')
parser.add_argument(
    '--val_img_txt',
    default=None,
    type=str,
    metavar='PATH',
    help='val images txt, samples should be listed as'
         'root/dog/xxx.png 0'
         '...'
         'root/cat/yyy.jpg 1')
parser.add_argument(
    '-w',
    '--work_dir',
    default='./',
    metavar='DIR',
    help='path to working folder')
parser.add_argument(
    '-a',
    '--arch',
    metavar='ARCH',
    default='resnet18',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) +
    ' (default: resnet18)')
parser.add_argument(
    "--model_kwargs",
    dest="model_kwargs",
    action=StoreDictKeyPair,
    nargs="+",
    metavar="KEY=VAL",
    help='additional hyper-parameters for model')

parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=90,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256), this is the total '
    'batch size of all GPUs on the current node when '
    'using Data Parallel or Distributed Data Parallel')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate',
    dest='lr')
parser.add_argument('--lr_step', default=30, type=float, dest='lr_step')
parser.add_argument('--cosine_annealing', dest='cosine_annealing', action='store_true', help='whether use cosine annealing')
parser.add_argument('--epochs_per_save', default=-1, type=int, dest='epochs_per_save', help='the interval to save ckpt')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--wd',
    '--weight-decay',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)',
    dest='weight_decay')
parser.add_argument(
        '--dropout', type=float, default=0.0)
parser.add_argument(
        '--feat_dim', type=int, default=0)
parser.add_argument(
    '-p',
    '--print-freq',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')

parser.add_argument('--retrain', dest='retrain', action='store_true')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument('--evaluate_aux', default=None, type=str)
parser.add_argument(
    '--evaluate_results_name',
    default='evaluate.result',
    type=str,
    help='name for saving the evaluate results')
parser.add_argument(
    '--evaluate_dataset', default='val', choices=['val', 'train'], type=str)
parser.add_argument(
    '--use-ten-crop-val', dest='ten_crop_val', action='store_true', help='use ten-crop in val')

parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--pretrained_model_path',
    default=None,
    type=str,
    help='model path of pretrained source.')
parser.add_argument(
    '--world-size',
    default=-1,
    type=int,
    help='number of nodes for distributed training')
parser.add_argument(
    '--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument(
    '--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument(
    '--data_seed', default=None, type=int, help='seed for loading data. ')
parser.add_argument(
    '--sampler_seed', default=None, type=int, help='seed for sampling data. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument(
    '--multiprocessing-distributed',
    action='store_true',
    help='Use multi-processing distributed training to launch '
    'N processes per node, which has N GPUs. This is the '
    'fastest way to use PyTorch for either single node or '
    'multi node data parallel training')
parser.add_argument(
    '--bct_loss_weight',
    default=0,
    type=float,
    help='loss weight for backward compatible representation learning')
parser.add_argument(
    '--bct_single_fc', default=False, action='store_true', help='use a single fc in bct')
parser.add_argument(
    '--bct_old_model',
    default=None,
    type=str,
    help='source model for backward compatible representation learning')
parser.add_argument(
    '--bct_eval_alpha',
    default=0,
    type=float,
    help='ensemble alpha in evaluation stage for bct model')
parser.add_argument(
    '--fc_chain',
    action='store_true',
    help='fully connected kd')
parser.add_argument(
    '--kd_model_arch',
    default=None,
    type=str,
    help='model architecture for knowledge distillation source.')
parser.add_argument(
    '--kd_model_path',
    default=None,
    nargs='+',
    type=str,
    help='model path of knowledge distillation source.')
parser.add_argument(
    '--kd_model_path_weight',
    default=None,
    nargs='+',
    type=float,
    help='weight for model path of knowledge distillation source.')
parser.add_argument(
    '--kd_model_output',
    default=None,
    type=str,
    help='model output for knowledge distillation source.')
parser.add_argument(
    '--kd_loss_weight', default=0, type=float, help='loss weight of KD loss ')
parser.add_argument(
    '--kd_temperature', default=5, type=float, help='temperature of KD loss')
parser.add_argument(
    '--cna_temperature', default=0.05, type=float, help='temperature of CNA loss')
parser.add_argument(
    '--kd_filter',
    default='all_pass',
    type=str,
    choices=['all_pass', 'neg_flip', 'old_correct', 'new_incorrect', 'likelihood'],
    help='the subset of training set applied KD loss')
parser.add_argument(
    '--kd_likelihood_source',
    default=None,
    type=str,
    help='likelihood source for weighting loss during knowledge distillation')
parser.add_argument(
    '--kd_model_num_classes',
    default=1000,
    type=int,
    help='kd model output class number')
parser.add_argument(
    '--ce_likelihood_source',
    default=None,
    type=str,
    help='likelihood source for weighting loss during cross entropy')
parser.add_argument('--ce-weight-base', default=1., type=float, help='the base weight of CE (when using ce_likelihood_source)')
parser.add_argument('--ce-weight-scale', default=1., type=float, help='the positive scale of CE (when using ce_likelihood_source)')
parser.add_argument('--kd-weight-base', default=1., type=float, help='the base weight of KD (when using kd_likelihood_source)')
parser.add_argument('--kd-weight-scale', default=1, type=float, help='the positive scale of KD (when using kd_likelihood_source)')
parser.add_argument(
    '--renyi_alpha', default=2, type=int, help='alpha in Renyi Divergence')
parser.add_argument(
    '--renyi_reorder', default=False, action='store_true', help='sort probabilities before calculating Renyi Divergence')
parser.add_argument(
    '--li_p', default=2, type=int, help='p-norm for LDI')
parser.add_argument(
    '--li_margin', default=1.0, type=float, help='margin in LDI')
parser.add_argument(
    '--li_one_sided', default=False, action='store_true', help='whether to inhibit one-side only')
parser.add_argument(
    '--li_compute_topk', default=-1, type=int, help='top-k classes with highest new logits values for inhibition (-1 means all classes)')
parser.add_argument(
    '--li_exclude_gt', default=True, action='store_true', help='whether to exclude gt class in LDI')
parser.add_argument(
    '--li_enhance_gt', default=False, action='store_true', help='whether to enhance (put more weights on) gt class in LDI')
parser.add_argument(
    '--li_enhance_gt_weight', default=0, type=float, help='loss weight for gt class in LDI (joint with --li-enhance-gt)')
parser.add_argument(
    '--li_margin_relative', default=False, action='store_true', help='whether to use relative margin')
parser.add_argument(
    '--li_use_p_norm', default=False, action='store_true', help='whether to calculate the norm in LDI')
parser.add_argument(
    '--rce_two_heads', default=False, action='store_true', help='whether to maintain two CE heads for RCE mode')
parser.add_argument(
    '--rp_dim_in_old', default=512, type=int, help='old models feature dim before random-projection')
parser.add_argument(
    '--rp_dim_in_new', default=512, type=int, help='new models feature dim before random-projection')
parser.add_argument(
    '--rp_dim_out', default=128, type=int, help='dim after random-projection')
parser.add_argument(
    '--rp_use_bias', default=False, action='store_true', help='whether to use bias when using random-projection')
parser.add_argument(
    '--rp_project_normalize', default=False, action='store_true', help='whether to normalize feature in random-projection')
parser.add_argument(
    '--rp_project_share', default=False, action='store_true', help='share parameters in old and new random projection matrix')
parser.add_argument(
    '--rp_project_new_frozen', default=False, action='store_true', help='freeze new models random projection matrix')
parser.add_argument(
    '--fa_margin', default=3.0, type=float, help='margin for FeatureAlignment')
parser.add_argument(
    '--fa_contrastive', default=False, action='store_true', help='whether to use contrastive version of FeatureAlignment')
parser.add_argument(
    '--kd_loss_mode',
    default='kl',
    type=str,
    choices=['kl', 'gt', 'l2', 'cna', 'renyi', 'li', 'rce', 'rp', 'fa', 'li+cna'],
    help='the supervision in knowledge distillation')
parser.add_argument(
    '--li_cna_alpha', default=1.0, type=float)
parser.add_argument(
    '--kd_alpha',
    default=None,
    type=float)
parser.add_argument(
    '--use_l2norm_fc',
    default=False,
    action='store_true')
parser.add_argument(
    '--l2norm_fc_t',
    default=0.01,
    type=float)
parser.add_argument(
    '--use_mimo',
    default=False,
    action='store_true')
parser.add_argument(
    '--mimo_m',
    default=1, type=int)
parser.add_argument(
    '--mimo_rho',
    default=0.6, type=float)
parser.add_argument(
    '--densenet-save-memory',
    default=False,
    action='store_true')
parser.add_argument(
    '--save_features',
    default=False, action='store_true')

# parser.add_argument('--aux_logits', dest='aux_logits', action='store_true')
parser.add_argument('--filter-base', default=0., type=float, help='the base weight of KD filter')
parser.add_argument('--filter-scale', default=1., type=float, help='the positive scale of KD filter')

parser.add_argument(
    '--auto-scale',
    action='store_true',
    help='auto scale learning rate and batch size by nodes')
parser.add_argument('--no-lr-scale',
                    action='store_true', help='set true to avoid scaling the learning rate')
parser.add_argument(
    '--auto-shotdown',
    action='store_true',
    help='auto shotdown after training. (Only active for 8-gpu servers)')
parser.add_argument(
    '--save-init-checkpoint',
    action='store_true',
    help='save model after initialization before training')
best_acc1 = 0

parser.add_argument(
    '--label_smoothing', default=0, type=float, help='epsilon for label smoothing')

parser.add_argument(
    '--use_bceloss', default=False, action='store_true')

parser.add_argument(
    '--train_linear_only', default=False, action='store_true')
parser.add_argument(
    '--reinit_linear', default=False, action='store_true')
parser.add_argument(
    '--use_cifar', default=True, type=bool)
parser.add_argument(
    '--load_from_s3', default=True, type=bool)
parser.add_argument(
    '--bucket_name', default=None, type=str, help='s3 bucket name for saving result')
parser.add_argument(
    '--desc', default=None, type=str, help='description for model')

def main():
    args = parser.parse_args()

    # scale batch_size
    ngpus_per_node = torch.cuda.device_count()
    if args.auto_scale:
        args.batch_size *= ngpus_per_node  # TODO: change to world size
        scale_factor = args.batch_size * args.world_size / 256 if not args.no_lr_scale else 1
        args.lr *= scale_factor
        args.workers *= ngpus_per_node

    if args.auto_shotdown and ngpus_per_node == 8 and os.environ.get("AUTO_SHUTDOWN") == 'TRUE':
        print('This server will shutdown after mission completion.', flush=True)

    print('\nStart Training From Configuration:')
    print(vars(args))

    if args.seed is not None:
        # random.seed(args.seed)
        # torch.manual_seed(args.seed)
        reset_seed(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

    if args.auto_shotdown and ngpus_per_node == 8 and os.environ.get("AUTO_SHUTDOWN") == 'TRUE':
        print('Mission Complete. Shutdown...', flush=True)
        os.system("sudo shutdown")


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    distributed_kwargs = dict()
    if args.model_kwargs == None:
        args.model_kwargs = dict()

    if args.seed is not None:
        print('Setting model seed to be {}'.format(args.seed))
        reset_seed(args.seed)

    # cudnn.benchmark = True

    # Data loading code
    if args.use_cifar:
        args.cosine_annealing = True
        import torchvision
        cls_num = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data, train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.CIFAR10(
            root=args.data, train=False, download=True, transform=transform
        )
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_trans_inception = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(299),
            transforms.ToTensor(),
            normalize,
        ])
        if args.train_img_txt is not None:
            if args.arch.startswith('inception'):
                train_dataset = ImageListDataset(img_list=args.train_img_txt,
                                                 transform=train_trans_inception,
                                                 folder=traindir)
            else:
                train_dataset = ImageListDataset(img_list=args.train_img_txt,
                                                 transform=train_trans,
                                                 folder=traindir)
            cls_num = train_dataset.class_num
        else:
            if args.arch.startswith('inception'):
                train_dataset = datasets.ImageFolder(
                    traindir,
                    train_trans_inception)
            else:
                train_dataset = datasets.ImageFolder(
                    traindir,
                    train_trans)
            cls_num = len([d.name for d in os.scandir(traindir) if d.is_dir()])

    if args.arch.startswith('googlenet') or args.arch.startswith('inception'):
        # model_kwargs['aux_logits'] = args.aux_logits
        args.model_kwargs['aux_logits'] = False

    if args.use_l2norm_fc:
        args.model_kwargs['norm_fc'] = True
        args.model_kwargs['norm_fc_t'] = args.l2norm_fc_t

    args.model_kwargs['dropout'] = args.dropout
    args.model_kwargs['feat_dim'] = args.feat_dim

    if args.use_mimo:
        args.model_kwargs['mimo_m'] = args.mimo_m
        args.model_kwargs['mimo_rho'] = args.mimo_rho

    if args.densenet_save_memory:
        args.model_kwargs['memory_efficient'] = True

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)
    # create model
    if args.bct_loss_weight == 0:
        if args.arch == 'swintrans':
            import timm
            model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, use_checkpoint=False)
        else:
            if 'StoDepth' in args.arch:
                model_func = getattr(stodepth_models, args.arch)
                distributed_kwargs['find_unused_parameters'] = True
            else:
                assert args.arch.startswith('resnet') or args.arch.startswith('densenet'), \
                       NotImplementedError("only support ResNet or DenseNet family")
                model_func = getattr(models, args.arch)

            if args.pretrained:
                print("=> using pre-trained model '{}'".format(args.arch))
                if args.pretrained_model_path is None:
                    model = model_func(pretrained=True, num_classes=cls_num, **(args.model_kwargs))
                else:
                    model = model_func(num_classes=cls_num, **(args.model_kwargs))
                    state_dict = torch.load(args.pretrained_model_path, map_location='cpu')
                    from collections import OrderedDict
                    c_state_dict = OrderedDict()
                    for key, value in state_dict.items():
                        if 'module.' in key:
                        # remove 'module.' of data parallel
                            name = key[7:]
                            c_state_dict[name] = value
                        else:
                            c_state_dict[key] = value
                    unfit_keys = model.load_state_dict(c_state_dict, strict=False)
                    print('=> these keys in model are not in state dict: {}'.format(unfit_keys.missing_keys))
                    print('=> these keys in state dict are not in model: {}'.format(unfit_keys.unexpected_keys))
                    print("=> loading done!")
            else:
                print("=> creating model '{}'".format(args.arch))
                model = model_func(num_classes=cls_num, **(args.model_kwargs))
    else:
        from networks import bct_models
        if args.bct_single_fc:
            args.model_kwargs['bct_single_fc'] = True
        print("=> creating model '{}' using bct".format(args.arch))
        print("=> old model: {}".format(args.bct_old_model if
                                        args.bct_old_model else 'torchvision'))
        print("=> pretrained: {}".format(args.pretrained))
        model = bct_models.__dict__[args.arch](
            pretrained=args.pretrained,
            old_model=args.bct_old_model,
            num_classes=cls_num,
            **(args.model_kwargs))

    if args.train_linear_only:
        print('=> train linear only')
        distributed_kwargs['find_unused_parameters'] = True
        assert args.pretrained or args.resume is not None, NotImplementedError("You need to specify args.pretrained when training linear only, otherwise the model would be nonsense")
    if args.reinit_linear:
        print('=> reinitialize linear layer')
        # reset_seed(args.seed)
        nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(model.fc.bias, 0)

    if args.kd_loss_weight > 0:
        args.kd_model_path = args.kd_model_path[0].split()
        print("=> using knowledge distillation.")
        if args.kd_model_output:
            kd_model = None
            kd_output = ModelOutputDataset(args.kd_model_output)
        elif args.kd_model_path and len(args.kd_model_path) == 1:
            if args.load_from_s3:
                p = args.kd_model_path[0]
                import boto3
                s3 = boto3.client('s3')
                with open(p, 'wb') as f:
                    s3.download_fileobj(args.bucket_name, p, f)
                print('downloaded {} from s3'.format(p))
            kd_model = models.__dict__[args.kd_model_arch](**(args.model_kwargs), num_classes=args.kd_model_num_classes)
            kd_state_dict = torch.load(args.kd_model_path[0], map_location=torch.device('cpu'))
            if 'state_dict' in kd_state_dict:
                kd_state_dict = kd_state_dict['state_dict']
            tmp = collections.OrderedDict()
            for k in kd_state_dict:
                tmp[k.split('module.')[-1]] = kd_state_dict[k]
            kd_model.load_state_dict(tmp)
            kd_output = None
        elif args.kd_model_path and len(args.kd_model_path) > 1:
            from networks.utils import EnsModel
            m_list = torch.nn.ModuleList()
            if args.fc_chain:
                kd_arch = [args.kd_model_arch for i in range(len(args.kd_model_path) - 1)] + ['resnet18']
            else:
                kd_arch = [args.kd_model_arch for i in range(len(args.kd_model_path))]
            for p, arch in zip(args.kd_model_path, kd_arch):
                if arch == 'swintrans':
                    m = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, use_checkpoint=False,
                                          num_classes=args.kd_model_num_classes)
                else:
                    m = models.__dict__[arch](**(args.model_kwargs), num_classes=args.kd_model_num_classes)
                if args.load_from_s3:
                    import boto3
                    s3 = boto3.client('s3')
                    with open(p, 'wb') as f:
                        s3.download_fileobj(args.bucket_name, p, f)
                    print('downloaded {} from s3'.format(p))
                kd_state_dict = torch.load(p, map_location=torch.device('cpu'))
                if 'state_dict' in kd_state_dict:
                    kd_state_dict = kd_state_dict['state_dict']
                tmp = collections.OrderedDict()
                for k in kd_state_dict:
                    tmp[k.split('module.')[-1]] = kd_state_dict[k]
                m.load_state_dict(tmp)
                m_list.append(m)
            if args.kd_model_path_weight is None:
                args.kd_model_path_weight = [1,] * len(m_list)
            assert(len(args.kd_model_path_weight) == len(m_list))
            kd_model = EnsModel(m_list, args.kd_model_path_weight)
            kd_output = None
        else:
            kd_model = models.__dict__[args.kd_model_arch](pretrained=True, **(args.model_kwargs))
            kd_output = None
    else:
        kd_model = None
        kd_output = None

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], **distributed_kwargs)
            if kd_model:
                kd_model.cuda(args.gpu)
                kd_model = torch.nn.parallel.DistributedDataParallel(
                    kd_model, device_ids=[args.gpu], **distributed_kwargs)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(
                model, **distributed_kwargs)
            if kd_model:
                kd_model.cuda()
                kd_model = torch.nn.parallel.DistributedDataParallel(
                    kd_model, **distributed_kwargs)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if kd_model:
            kd_model = kd_model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

        if kd_model:
            if args.kd_model_arch.startswith(
                    'alexnet') or args.kd_model_arch.startswith('vgg'):
                kd_model.features = torch.nn.DataParallel(kd_model.features)
                kd_model.cuda()
            else:
                kd_model = torch.nn.DataParallel(kd_model).cuda()

    # define loss function (criterion) and optimizer
    if args.label_smoothing == 0:
        if args.use_bceloss:
            assert args.kd_loss_weight == 0, NotImplementedError('use_bceloss=True does not allow kd_loss_weight > 0')
            criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
        else:
            criterion = nn.CrossEntropyLoss().cuda(args.gpu) if args.ce_likelihood_source is None else nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)
    else:
        from networks.utils import LabelSmoothingCrossEntropy
        criterion = LabelSmoothingCrossEntropy(epsilon=args.label_smoothing).cuda(args.gpu)

    fd_criterion = FocalDistillationLoss(focus_type=args.kd_filter,
                                         fd_alpha=args.filter_base,
                                         fd_beta=args.filter_scale,
                                         kl_temperature=args.kd_temperature,
                                         cna_temperature=args.cna_temperature,
                                         distillation_type=args.kd_loss_mode,
                                         renyi_alpha=args.renyi_alpha,
                                         renyi_reorder=args.renyi_reorder,
                                         li_p=args.li_p,
                                         li_margin=args.li_margin,
                                         li_one_sided=args.li_one_sided,
                                         li_exclude_gt=args.li_exclude_gt,
                                         li_compute_topk=args.li_compute_topk,
                                         li_enhance_gt=args.li_enhance_gt,
                                         li_enhance_gt_weight=args.li_enhance_gt_weight,
                                         li_margin_relative=args.li_margin_relative,
                                         li_use_p_norm=args.li_use_p_norm,
                                         rce_two_heads=args.rce_two_heads,
                                         rp_dim_in_old=args.rp_dim_in_old,
                                         rp_dim_in_new=args.rp_dim_in_new,
                                         rp_dim_out=args.rp_dim_out,
                                         rp_use_bias=args.rp_use_bias,
                                         rp_project_normalize=args.rp_project_normalize,
                                         rp_project_share=args.rp_project_share,
                                         rp_project_new_frozen=args.rp_project_new_frozen,
                                         fa_margin=args.fa_margin,
                                         fa_contrastive=args.fa_contrastive,
                                         li_cna_alpha=args.li_cna_alpha,
                                         ).cuda(args.gpu)

    if args.arch == 'swintrans':
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(
                fd_criterion.parameters()) if args.kd_loss_mode == 'rp' else model.parameters(),
            lr=0.001,
            weight_decay=0.05)
    else:
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(fd_criterion.parameters()) if args.kd_loss_mode == 'rp' else model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

    if args.cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                args.epochs,
                eta_min=args.lr * (0.1 ** (args.epochs / args.lr_step)))
    else:
        scheduler = None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if type(best_acc1) != int and args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if not args.evaluate and not args.retrain:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    if kd_output:
        train_dataset = CombineDataset([train_dataset, kd_output])
    if args.kd_likelihood_source is not None:
        kd_likelihood = NFLikelihoodDataset(args.kd_likelihood_source)
        train_dataset = CombineDataset([train_dataset, kd_likelihood])
    if args.ce_likelihood_source is not None:
        ce_likelihood = NFLikelihoodDataset(args.ce_likelihood_source)
        train_dataset = CombineDataset([train_dataset, ce_likelihood])

    if args.distributed:
        if torch.__version__ < '1.6':
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
        else:
            print('Setting sampler_seed to be {}'.format(args.sampler_seed))
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, seed=args.sampler_seed)
    else:
        train_sampler = None

    if args.ten_crop_val:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.TenCrop(224),
                    MultiCropToTensor(),
                    MultiCropNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])) if args.val_img_txt is None else
            ImageListDataset(
                img_list=args.val_img_txt,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.TenCrop(224),
                    MultiCropToTensor(),
                    MultiCropNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                folder=valdir),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)
    else:
        if args.use_cifar:
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=args.batch_size,
                                                     shuffle=False,
                                                     num_workers=args.workers,
                                                     pin_memory=True)
        else:
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(
                    valdir,
                    transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])) if args.val_img_txt is None else
                ImageListDataset(
                    img_list=args.val_img_txt,
                    transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]),
                    folder=valdir)
                ,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True)

    if args.evaluate:
        if args.evaluate_dataset == 'val':
            evaluate_loader = val_loader
        elif args.evaluate_dataset == 'train':
            evaluate_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(
                    traindir,
                    transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True)
        else:
            raise ValueError
        val_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        acc1, result = validate(evaluate_loader, model, val_criterion, args)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_result(result, args.evaluate_results_name, args.work_dir)
        return

    if args.save_init_checkpoint:
        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint(
                {
                    'epoch': 0,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': 0,
                    'optimizer': optimizer.state_dict(),
                    'fd_criterion': fd_criterion.state_dict() if args.kd_loss_mode == 'rp' else None,
                },
                False,
                work_dir=args.work_dir,
                filename='init.checkpoint.pth.tar')

    if args.data_seed is not None:
        print('Setting data_seed to be {}'.format(args.data_seed))
        reset_seed(args.data_seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    mixup_fn = None
    if args.arch == 'swintrans':
        from timm.scheduler.cosine_lr import CosineLRScheduler
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_loader) * args.epochs,
            t_mul=1.,
            lr_min=1e-5,
            warmup_lr_init=1e-6,
            warmup_t=len(train_loader) * 20,
            cycle_limit=1,
            t_in_epochs=False,
        )
        from timm.data import Mixup
        mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=cls_num)
        from timm.loss import SoftTargetCrossEntropy
        criterion = SoftTargetCrossEntropy()

    for epoch in range(args.start_epoch, args.epochs):
        print('Start Epoch {} at {}.'.format(
            epoch,
            datetime.datetime.now().isoformat()))
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.cosine_annealing:
            pass
        else:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            args,
            kd_model=kd_model,
            fd_criterion=fd_criterion,
            fix_backbone=args.train_linear_only,
            scheduler=scheduler,
            mixup_fn=mixup_fn)

        val_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        # evaluate on validation set
        acc1, result = validate(val_loader, model, val_criterion, args, fd_criterion=fd_criterion)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'fd_criterion': fd_criterion.state_dict() if args.kd_loss_mode == 'rp' else None,
                },
                is_best,
                work_dir=args.work_dir)
            if is_best:
                save_result(result, 'model_best.result', args.work_dir)
                if args.bucket_name is not None:
                    import boto3
                    s3 = boto3.client('s3')
                    # saving best output result to s3
                    if args.kd_loss_weight > 0:
                        best_result_name = 'model_{}_seed_{}_loss_{}_{}.result'.format(args.arch, args.seed,
                                                                                       args.kd_loss_mode, args.desc)
                    else:
                        best_result_name = 'model_{}_seed_{}.result'.format(args.arch, args.seed)
                    with open(os.path.join(args.work_dir, 'model_best.result'), 'rb') as f:
                        s3.upload_fileobj(f, args.bucket_name, best_result_name)
                    print('saved result to {}'.format(os.path.join(args.bucket_name, best_result_name)))
                    # saving best output model to s3
                    if args.kd_loss_weight > 0:
                        best_model_name = 'best_model_{}_seed_{}_loss_{}_{}.pth.tar'.format(args.arch, args.seed,
                                                                                            args.kd_loss_mode, args.desc)
                    else:
                        best_model_name = 'best_model_{}_seed_{}.pth.tar'.format(args.arch, args.seed)
                    with open(os.path.join(args.work_dir, 'model_best.pth.tar'), 'rb') as f:
                        s3.upload_fileobj(f, args.bucket_name, best_model_name)
                    print('saved model to {}'.format(os.path.join(args.bucket_name, best_model_name)))
            if (args.epochs_per_save > 0) and (epoch + 1) % args.epochs_per_save == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                        'fd_criterion': fd_criterion.state_dict() if args.kd_loss_mode == 'rp' else None,
                    },
                    False,
                    filename='checkpoint_epoch_{}.pth.tar'.format(epoch + 1),
                    work_dir=args.work_dir)
                save_result(result, 'model_epoch_{}.result'.format(epoch + 1), args.work_dir)



def train(train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          args,
          kd_model=None,
          fd_criterion=None,
          fix_backbone=False,
          scheduler=None,
          mixup_fn=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    kd_losses = AverageMeter('KD', ':6.3f')
    bct_losses = AverageMeter('BCT', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, kd_losses, bct_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    if fix_backbone:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False
            elif isinstance(module, nn.Conv2d):
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
            elif isinstance(module, nn.Linear):
                pass # here we assume that the whole only have fc at the heads

    end = time.time()
    for i, data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.ce_likelihood_source is not None:
            data, ce_likelihood = data
        if args.kd_likelihood_source is not None:
            data, kd_likelihood = data
        else:
            kd_likelihood = None
        if args.kd_model_output:
            data, kd_output = data

        images, target = data

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            if args.ce_likelihood_source is not None:
                ce_likelihood = ce_likelihood.cuda(args.gpu, non_blocking=True)
            if args.kd_likelihood_source is not None:
                kd_likelihood = kd_likelihood.cuda(args.gpu, non_blocking=True)
            if args.kd_model_output:
                kd_output = kd_output.cuda(args.gpu, non_blocking=True)
        if args.use_bceloss:
            # TODO: fix this hard-coded num_classes
            target_onehot = torch.zeros((target.size(0), 1000), dtype=torch.float)
            target_onehot[torch.arange(target.size(0)), target] = 1.
            target_onehot = target_onehot.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        target_for_acc = target

        if mixup_fn is not None:
            images, target = mixup_fn(images, target)

        # compute output
        if args.mimo_m > 1:
            images = torch.chunk(images, args.mimo_m, dim=0)
            target = torch.chunk(target, args.mimo_m, dim=0)
            if args.mimo_rho > 0:
                prob = torch.rand(images[0].shape[0])
                images[1][prob >= args.mimo_rho] = images[0][prob >= args.mimo_rho]
                target[1][prob >= args.mimo_rho] = target[0][prob >= args.mimo_rho]
            images = torch.cat(images, dim=1)
            target = torch.cat(target, dim=0)

        model_out = model(images)
        if isinstance(model_out, tuple):
            output, new_features = model_out
        else:
            output, new_features = model_out, None

        if args.mimo_m > 1:
            output = torch.chunk(output, args.mimo_m, dim=1)
            output = torch.cat(output, dim=0)
            new_features = torch.chunk(new_features, args.mimo_m, dim=1)
            new_features = torch.cat(new_features, dim=0)

        full_loss = 0

        if args.kd_loss_mode in ['cna', 'li+cna']:
            # CNA requires gathering to get the full batchsize
            all_new_features = all_gather_with_ag(new_features)
            full_new_features = torch.cat(all_new_features, dim=0)
            all_output = all_gather_with_ag(output)
            full_output = torch.cat(all_output, dim=0)
            all_target = all_gather_with_ag(target)
            full_target = torch.cat(all_target, dim=0)
        else:
            full_new_features = new_features
            full_output = output
            full_target = target

        if type(output).__name__ == 'MultibranchOutputs':
            output, aux_2, aux_1 = output
            full_loss += criterion(aux_2, target)
            full_loss += criterion(aux_1, target)

        if type(output).__name__ == 'GoogLeNetOutputs':
            output, aux_3, aux_2, aux_1 = output
#             full_loss += criterion(aux_4, target)
            full_loss += criterion(aux_3, target)
            full_loss += criterion(aux_2, target)
            full_loss += criterion(aux_1, target)

        # bct
        if type(output).__name__ == 'BCTOutputs':
            # old_output, output = output.parse()
            output, old_output = output #.parse()
            bct_loss = criterion(old_output, target) * args.bct_loss_weight
            bct_losses.update(bct_loss.item(), images.size(0))
            full_loss += bct_loss

        loss = criterion(output, target) if not args.use_bceloss else criterion(output, target_onehot)
        if args.ce_likelihood_source is not None:
            loss_weights = 1. / (args.ce_weight_scale * ce_likelihood + args.ce_weight_base)
            # loss = (loss * loss_weights).mean()
            loss = (loss * loss_weights).sum() / loss_weights.sum()
        full_loss += loss

        # knowledge distillation
        if args.kd_loss_weight > 0:

            # T = args.kd_temperature
            if not args.kd_model_output:
                kd_model.eval()
                kd_output_tp = kd_model(images)
                if isinstance(kd_output_tp, tuple):
                    kd_output, kd_features = kd_output_tp[0].detach(), kd_output_tp[1].detach()
                else:
                    kd_output, kd_features = kd_output_tp.detach(), None

                if args.kd_loss_mode in ['cna', 'li+cna']:
                    # CNA requires gathering to get the full batchsize
                    all_kd_features = all_gather_with_ag(kd_features)
                    full_kd_features = torch.cat(all_kd_features, dim=0)
                    all_kd_output = all_gather_with_ag(kd_output)
                    full_kd_output = torch.cat(all_kd_output, dim=0)
                else:
                    full_kd_features = kd_features
                    full_kd_output = kd_output
            else:
                full_kd_features = None
                full_kd_output = kd_output
            if kd_likelihood is not None:
                fd_loss = args.kd_loss_weight * fd_criterion(full_output, full_kd_output, target_for_acc, full_new_features, full_kd_features,
                                                             kd_likelihood=(args.kd_weight_scale * kd_likelihood + args.kd_weight_base))
            else:
                fd_loss = args.kd_loss_weight * fd_criterion(full_output, full_kd_output, full_target, full_new_features, full_kd_features)
            kd_losses.update(fd_loss.item(), images.size(0))

            # scale the loss by #world size because we used all gather.
            # In all_gather the gradients will be calculated repeatedly for #world_size times.

            if args.kd_alpha is None:
                full_loss += fd_loss
            else:
                full_loss = args.kd_alpha * fd_loss + (1 - args.kd_alpha) * full_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target_for_acc, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()
        if args.cosine_annealing:
            scheduler.step(epoch + i / len(train_loader))
        if args.arch == 'swintrans':
            scheduler.step_update(epoch * len(train_loader) + i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.distributed:
                if (dist.get_rank() == 0):
                    progress.display(i)
            else:
                progress.display(i)


def validate(val_loader, model, criterion, args, fd_criterion=None):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    kd_losses = AverageMeter('KD', ':6.3f')
    bct_losses = AverageMeter('BCT', ':6.3f')
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    pred = []
    gt = []
    outputs = []
    features = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if images.ndim == 5:
                bs = images.size(0)
                num_crop = images.size(1)
                images = images.flatten(0, 1)
            else:
                num_crop = 1
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if args.use_bceloss:
                # TODO: fix this hard-coded num_classes
                target_onehot = torch.zeros((target.size(0), 1000), dtype=torch.float)
                target_onehot[torch.arange(target.size(0)), target] = 1
                target_onehot = target_onehot.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.mimo_m > 1:
                images = images.repeat(args.mimo_m, 1, 1, 1)
                images = torch.chunk(images, args.mimo_m, dim=0)
                images = torch.cat(images, dim=1)
                output, ft = model(images)
                assert type(output).__name__ not in ['BCTOutputs', 'GoogLeNetOutputs', 'MultibranchOutputs'], \
                       'Currently args.use_mimo=True is only compatible with single output'
                output = torch.stack(output.chunk(args.mimo_m, dim=1))
                output = output.mean(0)
            else:
                if args.arch == 'swintrans':
                    output = model(images)
                else:
                    output, ft = model(images)
            if type(output).__name__ == 'BCTOutputs':
                # old_output, output = output
                output, old_output = output
                output = output * (1-args.bct_eval_alpha) + \
                    old_output*args.bct_eval_alpha
            elif type(output).__name__ == 'GoogLeNetOutputs':
                output, aux3, aux2, aux1 = output
                if args.evaluate_aux == 'avg':
                    output = (aux3 + aux2 + aux1 + output) / 4
                elif args.evaluate_aux == 'max':
                    output = torch.stack((aux3, aux2, aux1, output)).max(0)[0]
            elif type(output).__name__ == 'MultibranchOutputs':
                output, aux2, aux1 = output
                if args.evaluate_aux == 'avg':
                    output = (aux2 + aux1 + output) / 5
                elif args.evaluate_aux == 'max':
                    output = torch.stack((aux2, aux1, output)).max(0)[0]

            if num_crop > 1:
                output = output.view(bs, num_crop, -1)
                loss = criterion(output.mean(1), target) if not args.use_bceloss else criterion(output.mean(1), target_onehot)
                if args.ce_likelihood_source is not None:
                    loss = loss.mean()

                pred.append(torch.argmax(output.mean(1), 1).cpu())
                outputs.append(output.cpu())
                gt.append(target.cpu())

                if args.save_features:
                    ft = ft.view(bs, num_crop, -1)
                    features.append(ft.cpu())

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output.mean(1), target, topk=(1, 5))
            else:
                loss = criterion(output, target) if not args.use_bceloss else criterion(output, target_onehot)
                if args.ce_likelihood_source is not None:
                    loss = loss.mean()

                pred.append(torch.argmax(output, 1).cpu())
                outputs.append(output.cpu())
                gt.append(target.cpu())

                if args.save_features:
                    features.append(ft.cpu())

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
            top1=top1, top5=top5))

    info = dict(top1=top1.avg, top5=top5.avg)
    pred = torch.cat(pred, 0).cpu()
    gt = torch.cat(gt, 0).cpu()
    outputs = torch.cat(outputs, 0).cpu()

    if args.save_features:
        features = torch.cat(features, 0).cpu()
    if num_crop > 1:
        result = dict(info=info, pred=pred, gt=gt, outputs=outputs.mean(1), features=features, outputs_raw=outputs)
    else:
        result = dict(info=info, pred=pred, gt=gt, outputs=outputs, features=features)

    return top1.avg, result


def save_checkpoint(state,
                    is_best,
                    filename='checkpoint.pth.tar',
                    result=None,
                    work_dir='./'):
    filename = os.path.join(work_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(work_dir, 'model_best.pth.tar'))


def save_result(result, filename, work_dir):
    torch.save(result, os.path.join(work_dir, filename))
    print('save in {}'.format(filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // args.lr_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
