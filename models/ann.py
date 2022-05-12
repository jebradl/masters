import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_processing import FeatureDataset

feature_set = FeatureDataset('c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/pca95_data.csv')


class Net(nn.Module):
    def __init__(self):
        # defining functions for each linear layer
        super().__init__()
        self.fc1 = nn.Linear(207, 64) # in features, out features
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 8) # 8 genres

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)

        return F.softmax(x, dim=1)

net = Net()

train = feature_set.X_train, feature_set.y_train
test = feature_set.X_test, feature_set.y_test

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

loss_function = nn.CrossEntropyLoss()

optimiser = optim.SGD(net.parameters(), lr=0.001)

Epochs = 3

print("Training begins:")

for epoch in range(Epochs):
    for data in trainset:
        X, y = data
        net.zero_grad()
        print("X:", X.shape)
        output = net.forward(X)
        print(output.shape, y.shape, type(y))
        loss = loss_function(output,y)
        loss.backward()
        optimiser.step()

# correct = 0
# total = 0

# with torch.no_grad():
#     for data in testset:
#         X, y = data
#         output = net.forward(X.view(-1,28*28))
#         for idx, i in enumerate(output):
#             if torch.argmax(i) == y[idx]:
#                 correct += 1
#             total += 1
# print("accuracy:", round(correct/total, 3))