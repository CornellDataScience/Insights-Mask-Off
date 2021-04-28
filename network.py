import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,Dataset
from utils import *
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms


# Load the the dataset from raw image folders
training_csv = "data/trains.csv"
training_dir = "data"

siamese_dataset = SiameseDataset(training_csv,training_dir,transform = transforms.Compose([transforms.Resize((105,105)),transforms.ToTensor()]))
train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=4,
                        batch_size=2)

# Declare Siamese Network
net = SiameseNetwork()

# Declare Loss Function
criterion = ContrastiveLoss()

# Declare Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

# Set the device to cuda
device = torch.device('cpu')
model = train(train_dataloader, optimizer, criterion, net)
torch.save(model.state_dict(), "model.pt")
print("Model Saved Successfully")

testing_csv = "data/tests.csv"
testing_dir = "data"

# Load the test dataset
test_dataset = SiameseDataset(training_csv=testing_csv,training_dir=testing_dir,
                                transform=transforms.Compose([transforms.Resize((105,105)), transforms.ToTensor()]))

test_dataloader = DataLoader(test_dataset,num_workers=6,batch_size=1,shuffle=True)

# Test the network
count = 0
for i, data in enumerate(test_dataloader,0): 
    x0, x1 , label = data
    concat = torch.cat((x0,x1),0)
    output1,output2 = model(x0.to(device),x1.to(device))

    eucledian_distance = F.pairwise_distance(output1, output2)

    #if label==torch.FloatTensor([[0]]):
    #    label="Original Pair Of Signature"
    #else:
    #    label="Forged Pair Of Signature"
    
    if (abs(eucledian_distance.item()) < 0.5):
        label = "Same"
    else:
        label = "Different"

    #imshow(torchvision.utils.make_grid(concat))
    print("Predicted Eucledian Distance:-",eucledian_distance.item())
    print("Actual Label:-",label)
    count=count+1
    if count == 10:
        break
