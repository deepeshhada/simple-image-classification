import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 12, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(12, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(500, 250),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2),
            nn.Linear(250, 200),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(200, 84),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x
