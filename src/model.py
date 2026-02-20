import torch
import torch.nn as nn

class VirtualInfluencerNet(nn.Module):
    """
        Neural Network for Multi-Label Classification
        ReLU activation in hidden layers
        No Sigmoid here (handled in loss function for stability)
    """

    def __init__(self, input_size=36, hidden_size=64, output_size=4):
        super(VirtualInfluencerNet, self).__init__()

        # Sequential model define
        self.network = nn.Sequential(

            # First
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),                        # ReLU
            nn.BatchNorm1d(hidden_size),      # Normalize
            nn.Dropout(0.3),                  # Reduce overfitting

            # Second
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.3),

            # Output
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        """
        Forward propagation
        Returns raw logits (NOT probabilities)
        """
        return self.network(x)