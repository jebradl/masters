
import os
import numpy as np
from time import sleep
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter


dataset_path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/classified_small'
save_path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/saved_models/'

batch_size = 32  # 64 for gpu
img_height = 235
img_width = 352

epochs = 20 # 50
test_split = 0.8


transform = transforms.Compose([
            transforms.Resize([img_height, img_width]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

data = datasets.ImageFolder(root=dataset_path, transform=transform)
# print(len(data))

test_split_size = int(len(data)*test_split)
# print(test_split_size)
testset, valset = torch.utils.data.random_split(data, (test_split_size, len(data)-test_split_size))

test_ds = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
validation_ds = torch.utils.data.DataLoader(dataset=valset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=2)  # 3 input bc rgb?
        self.conv2 = nn.Conv2d(32, 64, 3, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=2)

        self.fc1 = nn.Linear(128*31*45, 128)
        self.fc2 = nn.Linear(128, 8)

        self.dropout = nn.Dropout2d(0.2) # do we need to add this in? eh why not

    def convs(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # print("2. x size:", x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        # print("3. x size:", x.size())
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        # print("4. x size:", x.size())

        x = self.dropout(x)

        return x

    def forward(self, x):
        x = self.convs(x)
        # print("5. x size:", x.size())
        x = x.view(-1, 128*31*45) # i still don't get the -1 but for later
        # print("6. x size:", x.size())

        x = self.fc1(x)
        # print("7. x size:", x.size())
        x = F.relu(x) 
        # print("8. x size:", x.size())
        x = self.fc2(x)
        # print("9. x size:", x.size())
        

        p = F.softmax(x, dim=1)
        
        return p

net = Net()
tb = SummaryWriter()


optimiser = optim.Adam(net.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

if __name__ == "__main__":
    for i in range(epochs):

        print("Epoch", i+1)
        
        total_loss = 0
        total_correct = 0

        with tqdm(test_ds, unit="batch") as tepoch:
            for data in tepoch:
                # print(len(data))
                X, y = data
                # print(X.shape, y)
                # print("1. x size:", X.shape)
                net.zero_grad()
                output = net.forward(X)

                loss = F.nll_loss(output, y)
                total_loss += loss.item()
                total_correct+= get_num_correct(output, y)

                loss.backward()
                optimiser.step()

        # tb.add_scalar("Loss", total_loss, i)
        # tb.add_scalar("Correct", total_correct, i)
        # tb.add_scalar("Accuracy", total_correct/ len(test_ds), i)

        # tb.add_histogram("conv1.bias", net.conv1.bias, i)
        # tb.add_histogram("conv1.weight", net.conv1.weight, i)
        # tb.add_histogram("conv2.bias", net.conv2.bias, i)
        # tb.add_histogram("conv2.weight", net.conv2.weight, i)
        # tb.add_histogram("conv2.bias", net.conv3.bias, i)
        # tb.add_histogram("conv2.weight", net.conv3.weight, i)

        print("total correct:", total_correct, "loss:", total_loss)

    # tb.close()


    correct = 0
    total = 0

    # data_loader = get_data_loader(test=False)
    with torch.no_grad():
        for data in validation_ds:
            X, y = data
            output = net.forward(X)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    print("accuracy:", round(correct/total, 3))

    torch.save(net, os.path.join(save_path, 'model_v1.pt'))