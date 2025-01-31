import math
import numpy as np
import torch
import torch.nn as nn

class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff
    """
    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, inputs):
        # Envelope function divided by r
        env_val = 1 / inputs + self.a * inputs**(self.p - 1) + self.b * inputs**self.p + self.c * inputs**(self.p + 1)

        return torch.where(inputs < 1, env_val, torch.zeros_like(inputs))

class DistanceLayer(nn.Module):
    """
    Projects a distance 0 < r < r_cut into an orthogonal basis of Bessel functions
    """
    def __init__(self, num_radial=8, r_cut=5.0, envelope_exponent=5):
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / r_cut
        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        self.frequencies = nn.Parameter(torch.tensor(np.pi * np.arange(1, num_radial + 1, dtype=np.float32)), requires_grad=True)

    def forward(self, inputs):
        # scale to range [0, 1]
        d_scaled = inputs * self.inv_cutoff

        # Necessary for proper broadcasting behaviour
        d_scaled = d_scaled.unsqueeze(-1)

        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * torch.sin(self.frequencies * d_scaled)

class FeedForwardLayer(nn.Module):
    """
    Convenience layer for defining a feed-forward neural network (a number of sequential dense layers)
    """
    def __init__(self, layer_sizes, layer_activations):
        super().__init__()

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if layer_activations[i] is not None:
                layers.append(layer_activations[i])
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)

class SimpleDistanceLayer(nn.Module):
    def __init__(self, r_cut=5.0):
        super().__init__()
        self.r_cut = r_cut

    def forward(self, dR):
        oodR = 1 / dR
        cosdR = (torch.cos(dR * math.pi / self.r_cut) + 1.0) / 2.0
        output = torch.stack([dR, oodR, cosdR], dim=1)
        return output
