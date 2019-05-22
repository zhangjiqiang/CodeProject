import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import NeuralNetModel
import json
import ImageProcess

def main():
    in_arg = get_input_args()
    train_data, trainloader, validloader, testloader = NeuralNetModel.loadData(in_arg.data_dir)
    train_model = NeuralNetModel.generateModel(in_arg.hidden_units, in_arg.arch)
    train_model, optimizer = NeuralNetModel.trainModel(train_model, trainloader, validloader, testloader,
                                                       in_arg.learning_rate, in_arg.epochs, in_arg.gpu)
    

    if len(in_arg.save_dir) > 0:
        NeuralNetModel.saveModel(in_arg.save_dir, train_model, train_data, optimizer, in_arg.epochs,
                                 in_arg.gpu, in_arg.arch)
def get_input_args():
    # Creates parse
    parser = argparse.ArgumentParser(description="train Neural model")
    parser.add_argument('data_dir', type=str,
                        help='path to folder of images')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16','alexnet','resnet18','densenet121'],
                        help='chosen model: you can choose:' + 'vgg16, alexnet, resnet18, densenet121')
    parser.add_argument('--save_dir', type=str, default='',
                        help='save model for example imagepth/checkpoint.pth')
    parser.add_argument('--gpu', action="store_true", help='chosen gpu')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learn rate')
    parser.add_argument('--hidden_units', nargs='+', type=int, help='hidden layer arch')
    parser.add_argument('--epochs', type=int, default=10, help='set step')
    # returns parsed argument collection
    return parser.parse_args()

if __name__ == "__main__":
    main()