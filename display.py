import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset
import utils
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import sys
import tensorflow as tf
import tkinter as tk
import tkinter.ttk as ttk

THRESHOLD = 0.16


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(

            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2))

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


def are_same(im1, im2):
    img1 = Image.open(im1)
    img2 = Image.open(im2)

    transform = transforms.Compose(
        [transforms.Resize((105, 105)), transforms.ToTensor()])

    # Loading the image

    img1 = img1.convert("L")
    img2 = img2.convert("L")
    # Apply image transformations
    img1 = transform(img1)
    img2 = transform(img2)

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork()
    model.load_state_dict(torch.load('model2.pt'))

    # model.eval()

    # Gives you the feature vector of both inputs
    output1, output2 = model(img1.to(device), img2.to(device))
    # Compute the distance
    euclidean_distance = F.pairwise_distance(output1, output2)
    # with certain threshold of distance say its similar or not
    print(euclidean_distance.item())
    if(abs(euclidean_distance.item()) > THRESHOLD):
        print("different")
    else:
        print("same")

    return euclidean_distance <= THRESHOLD  # true -> same person


if __name__ == '__main__':
    try:
        img1 = sys.argv[1]
        img2 = sys.argv[2]

        are_same(img1, img2)

        # if data not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    except IndexError:
        print('Error: Provide an input folder')
