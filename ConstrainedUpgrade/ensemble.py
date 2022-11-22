import sys
import argparse
from analysis.utils import ModelAnalizer
import torch

parser = argparse.ArgumentParser()
parser.add_argument('target')
parser.add_argument('source', nargs='+')
args = parser.parse_args()


if __name__ == "__main__":
    target = 0
    for s in args.source:
        m = ModelAnalizer(s)
        print(s, m.Acc())
        target += m
        del m
    print(args.target, target.Acc())
    target.dump(args.target)


