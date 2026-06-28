# model.py
# ANN architecture definition - must match training exactly
# This is needed to load the saved weights correctly

import torch.nn as nn

class AQI_ANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AQI_ANN, self).__init__()
        
        self.network = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Hidden Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Hidden Layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output Layer
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)