import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler


class FeatureDataset(Dataset):

    def __init__(self, file_name):
        super().__init__()
        print("dataset")

        # reading csv and loaidng data
        dataset_ = pd.read_csv(file_name)
        x = dataset_.iloc[1:8001, 1:208].values
        y = dataset_.iloc[1:8001, 208].values

        # feature scaling
        sc = StandardScaler()
        x_ = sc.fit_transform(x)
        # y_train = y

        cats = np.unique(y)

        print("genre values:", cats)

        for i, cat in enumerate(cats):
            y[y==cat] = -i

        y *= -1

        print("y_train shape:", np.unique(y))

        x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.3)

        # convert to torch tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.int64)
        self.X_test = torch.tensor(x_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.int64)
        # self.y_train = torch.tensor(torch.nn.functional.one_hot(torch.tensor(y)), dtype=torch.int64)

        print(self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape)

    
    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx], self.X_test[idx], self.y_test[idx]