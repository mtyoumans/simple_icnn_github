"""Provides a Class Implementing a standard MLP 
(Multi-Layer Perceptron)

    Typical usage:
    model = MLP()
"""

from torch import nn


class MLP(nn.Module):
    """Implements standard MLP (Multi-Layer Perceptron)

    Class for standard MLP or feedforward neural network
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
