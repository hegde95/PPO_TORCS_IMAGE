import torch
import torch.nn as nn
from torch.distributions import Normal

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # self.relu1 = nn.ReLU()

        # self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        # self.relu2 = nn.ReLU()

        # self.pool = nn.MaxPool2d(kernel_size=2)

        # self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        # self.relu3 = nn.ReLU()

        # self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        # self.relu4 = nn.ReLU()

        # self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=30)
        
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=16 * 16 * 24 * 4, out_features=num_inputs)
        )

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):
        value = self.critic(self.cnn_layers(x))
        mu = self.actor(self.cnn_layers(x))
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value
