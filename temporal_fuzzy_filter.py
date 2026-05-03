import torch
import torch.nn as nn

class TemporalFuzzyFilter(nn.Module):
    def __init__(self, feature_num=80, time_steps_num=33):
        super(TemporalFuzzyFilter, self).__init__() 
        self.feature_num = feature_num
        self.time_steps_num = time_steps_num
        self.rule_num = 10
        
        # Notice these map to feature_num (80), whereas the spatial filter mapped to time_steps_num (33)
        self.matrix_mu = nn.Parameter(torch.rand(self.rule_num, self.feature_num) * 0.1)
        self.matrix_sigma = nn.Parameter(torch.rand(self.rule_num, self.feature_num) + 0.1)
        self.W_Q = nn.Parameter(torch.randn(self.rule_num, self.feature_num, self.feature_num) * 0.01)
        self.W_V = nn.Parameter(torch.randn(self.rule_num, self.feature_num, self.feature_num) * 0.01)

    def forward(self, input_data):
        # input_data shape from the spatial filter: (Batch, 33, 80) -> Batch, Time, Features
        
        # 1. Compute W_Q * signal for ALL time steps and rules instantly!
        # 'btf' = Batch, Time, Features. 'rof' = Rules, Out_Features, In_Features
        wq_dot_signals = torch.einsum('btf, rof -> btro', input_data, self.W_Q) # Shape: (Batch, 33, 10, 80)
        
        # 2. Reshape mu and sigma to apply to all 33 time steps at once
        mu = self.matrix_mu.view(1, 1, self.rule_num, self.feature_num)
        sigma = self.matrix_sigma.view(1, 1, self.rule_num, self.feature_num)
        
        # 3. Calculate Gaussian Fire Strength
        squared_distance = ((wq_dot_signals - mu) / sigma) ** 2
        fire_strength = torch.exp(-0.5 * torch.sum(squared_distance, dim=-1)) # Shape: (Batch, 33, 10)
        
        # 4. Softmax Normalization
        cumulative_fire = torch.sum(fire_strength, dim=2, keepdim=True)
        softmax_value = fire_strength / (cumulative_fire + 1e-8) # Shape: (Batch, 33, 10)
        
        # 5. Compute W_V * signal for ALL time steps and rules instantly
        wv_dot_signals = torch.einsum('btf, rof -> btro', input_data, self.W_V) # Shape: (Batch, 33, 10, 80)
        
        # 6. Apply Softmax weights to the W_V signals
        weighted_output = softmax_value.unsqueeze(-1) * wv_dot_signals
        
        # 7. Sum across rules to get the final temporal output
        temporal_output = torch.sum(weighted_output, dim=2) # Shape: (Batch, 33, 80)
        
        return temporal_output