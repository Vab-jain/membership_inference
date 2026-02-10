# define the different types of attack models here

from torch import nn


class BasicNN(nn.Module):
    def __init__(self, in_features, hidden_features=64, out_features=2) -> None:
        super(BasicNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(hidden_features, out_features)
        )
            
    def forward(self, x):
        return self.network(x)

class BasicNN_v2(nn.Module):
    def __init__(self, in_features, hidden_features=64, out_features=2) -> None:
        super(BasicNN_v2, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_features*2),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(hidden_features*2, hidden_features*4),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(hidden_features*4, out_features)
        )
    
    def forward(self, x):
        return self.network(x)

# class BasicCNN(nn.Module):
#     def __init__(self, in_features, hidden_features=64, out_features=2) -> None:
#         super(BasicNN, self).__init__()

#         self.network = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5),
#             nn.Linear(in_features, hidden_features),
#             nn.Dropout(),
#             nn.Tanh(),
#             nn.Linear(hidden_features, hidden_features*2),
#             nn.Tanh(),
#             nn.Linear(hidden_features*2, out_features)
#         )
    
#     def forward(self, x):
#         return self.network(x)