import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=6, hidden_sizes=(64, 64), activation_fn=nn.ELU):
        super().__init__()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        predictions = self.network(x)
        return predictions