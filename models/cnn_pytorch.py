
import os
import numpy as np
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# from torch.utils.tensorboard import SummaryWriter


dataset_path = 'C:/Users/jeb1618/masters/models/data/fma/classified_small'
save_path = 'C:/Users/jeb1618/masters/models/saved_models/'

batch_size = 64  # 64 for gpu
img_height = 235
img_width = 352

epochs = 50 # 50
test_split = 0.8

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []
x_epoch = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./lossGraphs', 'train.jpg'))


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
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        x = self.dropout(x)

        return x

    def forward(self, x):

        x = self.convs(x)
        x = x.view(-1, 128*31*45)

        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)

        p = F.softmax(x, dim=1)
        
        return p

net = Net()
# tb = SummaryWriter()


optimiser = optim.Adam(net.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


if __name__ == "__main__":
    for i in range(epochs):

        print("Epoch", i+1)
        
        running_loss = 0
        running_correct = 0

        with tqdm(test_ds, unit="batch") as tepoch:
            for data in tepoch:
                # print(len(data))
                X, y = data
                # print(X.shape, y)
                # print("1. x size:", X.shape)
                net.zero_grad()
                output = net.forward(X)

                loss = F.nll_loss(output, y)
                running_loss += loss.item() * batch_size
                running_correct += get_num_correct(output, y)

                loss.backward()
                optimiser.step()

        epoch_loss = running_loss / batch_size
        epoch_acc = running_correct / batch_size
        # y_loss['train'].append(epoch_loss)
        # y_err['train'].append(1.0 - epoch_acc)

        print("total correct:", running_correct, "accuracy", epoch_acc, "loss:", epoch_loss)

        # if i == len(range(epochs))-1:
        #     draw_curve(i)

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
    
    # draw_curve()
    torch.save(net, os.path.join(save_path, 'model_v2.pt'))