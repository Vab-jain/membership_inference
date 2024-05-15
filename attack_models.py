# define the different types of attack models here

from torch import nn


class BasicNN(nn.Module):
    def __init__(self, in_features, hidden_features=64, out_features=2) -> None:
        super(BasicNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features)
        )
    
    def forward(self, x):
        return self.network(x)