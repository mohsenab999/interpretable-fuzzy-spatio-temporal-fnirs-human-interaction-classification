import torch
import torch.nn as nn

class SpatialFuzzyFilter(nn.Module):
    def __init__(self, feature_num=80, time_steps_num=33):
        super(SpatialFuzzyFilter, self).__init__()
        self.feature_num = feature_num
        self.time_steps_num = time_steps_num
        self.rule_num = 10
        
        self.matrix_mu = nn.Parameter(torch.rand(self.rule_num, self.time_steps_num) * 0.1)
        self.matrix_sigma = nn.Parameter(torch.rand(self.rule_num, self.time_steps_num) + 0.1)
        self.W_Q = nn.Parameter(torch.randn(self.rule_num, self.time_steps_num, self.time_steps_num) * 0.01)
        self.W_V = nn.Parameter(torch.randn(self.rule_num, self.time_steps_num, self.time_steps_num) * 0.01)

    def forward(self, input_data):
        # input_data shape: (Batch, 80, 33) 
        
        # 1. Compute W_Q * signal for ALL channels and rules instantly!
        # 'bft' = Batch, Features, Time. 'rot' = Rules, Out_Time, In_Time
        wq_dot_signals = torch.einsum('bft, rot -> bfro', input_data, self.W_Q) # Shape: (Batch, 80, 10, 33)
        
        # 2. Reshape mu and sigma so PyTorch can apply them to all 80 channels at once
        mu = self.matrix_mu.view(1, 1, self.rule_num, self.time_steps_num)
        sigma = self.matrix_sigma.view(1, 1, self.rule_num, self.time_steps_num)
        
        # 3. Calculate Gaussian Fire Strength
        squared_distance = ((wq_dot_signals - mu) / sigma) ** 2
        fire_strength = torch.exp(-0.5 * torch.sum(squared_distance, dim=-1)) # Shape: (Batch, 80, 10)
        
        # 4. Softmax Normalization
        cumulative_fire = torch.sum(fire_strength, dim=2, keepdim=True)
        softmax_value = fire_strength / (cumulative_fire + 1e-8) # Shape: (Batch, 80, 10)
        
        # 5. Compute W_V * signal for ALL channels and rules instantly
        wv_dot_signals = torch.einsum('bft, rot -> bfro', input_data, self.W_V) # Shape: (Batch, 80, 10, 33)
        
        # 6. Apply Softmax weights to the W_V signals
        weighted_output = softmax_value.unsqueeze(-1) * wv_dot_signals
        
        # 7. Sum across rules to get the final spatial output
        spatial_output = torch.sum(weighted_output, dim=2) # Shape: (Batch, 80, 33)
        
        # 8. Transpose for the Temporal Filter (Batch, Time, Features)
        temporal_input = spatial_output.permute(0, 2, 1) 
        
        return temporal_input