import torch.nn as nn


class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 2000),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(2000, 2000),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.3),
            nn.Linear(2000, 2000),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(2000, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        return self.net(X)
