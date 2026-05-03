import torch
import torch.nn as nn
from torch_spatial_IT2_fuzzy_filter import spatialIT2fuzzyfilter
from torch_temporal_IT2_fuzzy_filter import temporalIT2fuzzyfilter


class IT2SpatioTemporalBCINetwork(nn.Module):
    
    def __init__(self):
        super(IT2SpatioTemporalBCINetwork, self).__init__()
        
        # 1. Initialize your custom Fuzzy layers
        self.spatial_layer = spatialIT2fuzzyfilter(feature_num=80, time_steps_num=33)
        self.temporal_layer = temporalIT2fuzzyfilter(feature_num=80, time_steps_num=33)
        
        # 2. Add a standard neural network layer for the final classification
        # The input is 33 * 80 = 2640 (the size of your flattened fuzzy output)
        # The output is 1 (a single number predicting the class)
        # Replace your old self.classifier with this:
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2640, out_features=32),
            nn.ReLU(), # Adds non-linear thinking capacity
            nn.Dropout(p=0.2), # Drops 60% of connections randomly to prevent overfitting
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=16, out_features=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = x.permute(0, 2, 1)
        
        x = self.spatial_layer(x) 

        x = x.permute(0, 2, 1)
        

        x = self.temporal_layer(x) 

        x = torch.flatten(x, start_dim=1) 
        
     
        x = self.classifier(x)
        
        
        prediction = self.sigmoid(x)
        
        return prediction