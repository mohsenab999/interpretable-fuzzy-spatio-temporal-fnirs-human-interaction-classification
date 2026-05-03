import torch
import torch.nn as nn
from spatial_fuzzy_filter import SpatialFuzzyFilter
from temporal_fuzzy_filter import TemporalFuzzyFilter

class SpatioTemporalBCINetwork(nn.Module):
    def __init__(self):
        super(SpatioTemporalBCINetwork, self).__init__()
        
        # 1. Initialize your custom Fuzzy layers
        self.spatial_layer = SpatialFuzzyFilter(feature_num=80, time_steps_num=33)
        self.temporal_layer = TemporalFuzzyFilter(feature_num=80, time_steps_num=33)
        
        # 2. Add a standard neural network layer for the final classification
        # The input is 33 * 80 = 2640 (the size of your flattened fuzzy output)
        # The output is 1 (a single number predicting the class)
        # Replace your old self.classifier with this:
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2640, out_features=64),
            nn.ReLU(), # Adds non-linear thinking capacity
            nn.Dropout(p=0.5), # Drops 50% of connections randomly to prevent overfitting
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=32, out_features=1)
        )
        
        # 3. Add a Sigmoid function to squeeze the output between 0.0 and 1.0
        # (Close to 0 = No Touch, Close to 1 = Hand-Holding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      
        x = self.spatial_layer(x) 
        
  
        x = self.temporal_layer(x) 
    
        x = torch.flatten(x, start_dim=1) 
        
     
        x = self.classifier(x)
        

        prediction = self.sigmoid(x)
        
        return prediction