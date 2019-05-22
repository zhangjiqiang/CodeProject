import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import numpy as np
import json
import ImageProcess
def main():
    print(torch.cuda.is_available())
    in_arg = get_input_args()
    category_names_path = in_arg.category_names
    category_names = None
    if len(category_names_path) > 0:
        with open(category_names_path, 'r') as f:
            category_names = json.load(f)

    ps_K, flower_tpye = ImageProcess.predict(in_arg.input_path, in_arg.checkpoint, category_names,
                                             in_arg.top_k, in_arg.gpu)
    print(ps_K)
    print(flower_tpye)


def get_input_args():
    # Creates parse
    parser = argparse.ArgumentParser(description="predict flower type")
    parser.add_argument('input_path', type=str,
                        help='path of images for predicting')
    parser.add_argument('checkpoint', type=str,
                        help='load model path')
    parser.add_argument('--gpu', action="store_true", help='chosen gpu')
    parser.add_argument('--top_k', type=int, default=1, help='return max K class')
    parser.add_argument('--category_names', type=str, default='', help='map type to name')
    # returns parsed argument collection
    return parser.parse_args()

if __name__ == "__main__":
    main()