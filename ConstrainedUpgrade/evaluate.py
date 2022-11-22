import sys
import argparse
from analysis.utils import ModelAnalyzer
import torch

parser = argparse.ArgumentParser()
parser.add_argument('old_model')
parser.add_argument('new_model', nargs='+')
parser.add_argument('--torchvision_info', default='./work_dir/torchvision_models/torchvision.model_info.cache')
args = parser.parse_args()

if __name__ == "__main__":
    try:
        old_model = ModelAnalyzer(args.old_model)
    except:
        torchvision_info = torch.load(args.torchvision_info)
        old_model = ModelAnalyzer(torchvision_info[args.old_model])


    print('Old Model: {}'.format(args.old_model))
    print('\tAcc: {:.4f}'.format(old_model.Acc()))
    for i in args.new_model:
        try:
            new_model = ModelAnalyzer(i)
        except:
            torchvision_info = torch.load(args.torchvision_info)
            new_model = ModelAnalyzer(torchvision_info[i])
        print('New Model: {}'.format(i))
        print('\tAcc: {:.4f}'.format(new_model.Acc()))
        print('\tNFR: {:.4f}'.format(new_model.NFR(old_model)))
        print('\tPFR: {:.4f}'.format(new_model.PFR(old_model)))
