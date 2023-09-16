import torch
import torch.nn as nn


class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            # self.fc2 = nn.Linear(128, 10)
            self.fc2 = nn.Linear(128, 2) # fake or real 분류를 위해 이진 분류

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x