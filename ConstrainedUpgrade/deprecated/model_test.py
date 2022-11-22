import argparse
import os
import time
import torch
import torchvision
from torchvision import datasets, transforms as T
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

parser = argparse.ArgumentParser(description='ImageNet Test')
parser.add_argument('--data', metavar='DIR', default="~/resource/imagenet",
                    help='path to dataset')
parser.add_argument('--cache', default='../work_dir/model_predictions/model_info.cache', help='path to cache')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

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
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    pred = []
    gt = []
    outputs = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            pred.append(torch.argmax(output, 1))
            outputs.append(output)
            gt.append(target)
            loss = criterion(output, target)

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
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    info = dict(top1=top1.avg, top5=top5.avg)
    pred = torch.cat(pred, 0)
    gt = torch.cat(gt, 0)
    outputs = torch.cat(outputs, 0)
    return info, pred, gt, outputs


def test_pretrain_model(model_name, args, arch=None, model_path=None):

    print('Process:{}'.format(model_name))
    if arch is None:
        arch = model_name

    if 'StoDepth' in arch:
        import TYY_stodepth_lineardecay as stodepth_models
        model = getattr(stodepth_models, arch)(pretrained=True)
    else:
        model = getattr(models, arch)(pretrained=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    if model_name.startswith('alexnet') or model_name.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    if model_path:
        state_dict = model.state_dict()
        input_state_dict = torch.load(model_path)['state_dict']
        if 'state_dict' in input_state_dict:
            input_state_dict = input_state_dict['state_dict']
        for k in input_state_dict:
            if k in state_dict:
                state_dict[k] = input_state_dict[k]
            else:
                print('Unexpected key {} in state_dict.'.format(k))
        model.load_state_dict(state_dict)
    return validate(val_loader, model, criterion.cuda(args.gpu), args)

if __name__ == '__main__':

    args = parser.parse_args()
    architectures = [
        'resnet18_StoDepth_lineardecay',
        'resnet34_StoDepth_lineardecay',
        'resnet50_StoDepth_lineardecay',
        'resnet101_StoDepth_lineardecay',
        'resnet152_StoDepth_lineardecay'
        # 'alexnet', 
        # 'vgg11', 
        # 'vgg13', 
        # 'vgg16', 
        # 'vgg19', 
        # 'vgg11_bn', 
        # 'vgg13_bn', 
        # 'vgg16_bn', 
        # 'vgg19_bn', 
        # 'resnet18', 
        # 'resnet34', 
        # 'resnet50', 
        # 'resnet101', 
        # 'resnet152', 
        # 'resnext50_32x4d',
        # 'resnext101_32x8d',
        # 'wide_resnet50_2',
        # 'wide_resnet101_2',
        # 'squeezenet1_0', 
        # 'squeezenet1_1', 
        # 'densenet121', 
        # 'densenet169', 
        # 'densenet201', 
        # 'densenet161', 
        # 'inception_v3', 
        # 'googlenet', 
        # 'shufflenet_v2_x0_5', 
        # 'shufflenet_v2_x1_0', 
        # 'mobilenet_v2', 
        # 'mnasnet0_5', 
        # 'mnasnet1_0', 
        # ('resnet18[seed:?]', 'resnet18', '../work_dir/torchvision_models/resnet18|seed:?/model_best.pth.tar'),
        # ('resnet18[seed:2]', 'resnet18', '../work_dir/torchvision_models/resnet18|seed:2/model_best.pth.tar'),
        # ('resnet18[seed:3]', 'resnet18', '../work_dir/torchvision_models/resnet18|seed:3/model_best.pth.tar'),
        # ('resnet18[seed:3]', 'resnet18', '../work_dir/torchvision_models/resnet18|seed:3/model_best.pth.tar'),
        # ('resnet18[bct:resnet18[seed:2]]', 'resnet18', '../work_dir/torchvision_models/resnet18_bct_old.seed2/model_best.pth.tar'),
        # ('resnet18[bct:resnet18]', 'resnet18', '../work_dir/torchvision_models/resnet18_bct_old.torchvision/model_best.pth.tar'),
        # ('resnet18[kd:resnet18[seed:2]]', 'resnet18', '../work_dir/torchvision_models/resnet18_kd.renet18_seed.2/model_best.pth.tar'),
        # ('resnet18[kd:resnet18]', 'resnet18', '../work_dir/torchvision_models/resnet18[kd:resnet18]/model_best.pth.tar'),
        # ('resnet18[kd:resnet50]', 'resnet18', '../work_dir/torchvision_models/resnet18_50_kd/model_best.pth.tar'),
        # ('resnet50[kd:resnet18]', 'resnet50', '../work_dir/torchvision_models/resnet50_18_kd/model_best.pth.tar'),
        # ('resnet50[kd:resnet50]', 'resnet50', '../work_dir/torchvision_models/resnet50_kd/model_best.pth.tar'),
        ]
    if (args.cache is not None) and (os.path.exists(args.cache)):
        model_info = torch.load(args.cache)
    else:
        model_info = dict()


    for m in architectures:
        if type(m) == tuple or type(m) == list:
            m, arch, path = m
            info, pred, gt, outputs = test_pretrain_model(m, args, arch, path)
            model_info[m] = dict(info=info, pred=pred, gt=gt, outputs=outputs)
        else:
            info, pred, gt, outputs = test_pretrain_model(m, args)
            model_info[m] = dict(info=info, pred=pred, gt=gt, outputs=outputs)
    if args.cache is not None:
        torch.save(model_info, args.cache)
    print(model_info)
