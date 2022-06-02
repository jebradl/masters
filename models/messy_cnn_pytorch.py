import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

### model parameters

dataset_path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/classified_small'
dataset_dir = os.listdir(dataset_path)

batch_size = 32  # 64 for gpu
img_height = 235
img_width = 352

epochs = 20 # 50
# val_split = 0.2

data = datasets.ImageFolder(root=data_dir, transform=transform)
loader = torch.utils.data.DataLoader()

def get_data_loader(data_dir=dataset_dir, batch_size=32, train = True):
    
    transform = {
        'train': transforms.Compose([
            transforms.Resize([224,224]), # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.RandomHorizontalFlip(), # Flip the data horizontally
            #TODO if it is needed, add the random crop
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    }

    data = datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader


# The way to get one batch from the data_loader
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    data_loader = get_data_loader()

    for i in range(10):
        batch_x, batch_y = next(iter(data_loader))
        print(np.shape(batch_x), batch_y)



class Net(nn.Module):
    def __init__(self):
        # defining functions for each linear layer
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2) # input, output, convolutional kernel dimensions
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2) # padding is to prevent dataloss

        self.fc1 = nn.Linear(64*7*7, 128) # in features, out features - 7*7 is max pooling
        self.fc2 = nn.Linear(128, 10) 


    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))  # applies the max funtion over a 2x2 patch
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))

        return x


    def forward(self, x):
        x = self.convs(x) # actuates convolutional layers
        x = x.view(-1, 64*7*7) # resizes data for processing without errors, forces resized tensor to have same size as input

        x = F.relu(self.fc1(x)) # combines outputs of convolutional layers and uses classifier structure to get probability distribution for input image
        x = self.fc2(x)

        return F.softmax(x, dim=1)

net = Net()
