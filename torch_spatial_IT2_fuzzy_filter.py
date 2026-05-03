import torch
import torch.nn.functional as F
import torch.nn as nn

class spatialIT2fuzzyfilter(nn.Module):
    def __init__(self, feature_num=80, time_steps_num=33):
        super(spatialIT2fuzzyfilter, self).__init__()
        self.feature_num = feature_num
        self.time_steps_num = time_steps_num
        self.rule_num = 10
        
        
        # --- Antecedent Parameters (Uncertain Mean IT2) ---
        # Notice these are now sized to self.feature_num (80)
        self.matrix_mu_antecedent_first = nn.Parameter(torch.randn(self.rule_num, self.feature_num) * 0.1)
        self.matrix_mu_antecedent_second = nn.Parameter(torch.randn(self.rule_num, self.feature_num) * 0.1)
        self.matrix_sigma_antecedent = nn.Parameter(torch.ones(self.rule_num, self.feature_num))
        self.c1_consequent = nn.Parameter(torch.randn(self.rule_num, 1) * 0.1)
        self.c2_consequent = nn.Parameter(torch.randn(self.rule_num, 1) * 0.1)
        
        
        # --- TSK Matrices ---
        # Sized to (10, 80, 80)
        self.W_Q = nn.Parameter(torch.randn(self.rule_num, self.feature_num, self.feature_num) * 0.1)
        self.W_V = nn.Parameter(torch.randn(self.rule_num, self.feature_num, self.feature_num) * 0.1)

    def forward(self, input_data):
        # input_data shape from the network layer: (Batch, Time, Features) -> (Batch, 33, 80)
        
        # =========================================================
        # 1. ANTECEDENT QUERY (W_Q * X)
        # =========================================================
        # 'btf': Batch, Time, Feature_in
        # 'rcf': Rule, Feature_out, Feature_in
        # Result: 'btrc' -> (Batch, 33, 10, 80)
        query = torch.einsum('btf, rcf -> btrc', input_data, self.W_Q)
        
        # =========================================================
        # 2. RESHAPE PARAMETERS FOR BROADCASTING
        # =========================================================
        # Broadcasting to match (Batch, 33, 10, 80)
        mu_1 = self.matrix_mu_antecedent_first.view(1, 1, self.rule_num, self.feature_num)
        mu_2 = self.matrix_mu_antecedent_second.view(1, 1, self.rule_num, self.feature_num)
        sig_a = torch.abs(self.matrix_sigma_antecedent).view(1, 1, self.rule_num, self.feature_num) + 1e-4
        
        mu_lower = torch.min(mu_1, mu_2)
        mu_upper = torch.max(mu_1, mu_2)
        
        # =========================================================
        # 3. CALCULATE IT2 DISTANCES 
        # =========================================================
        query_clamped = torch.clamp(query, min=mu_lower, max=mu_upper)
        dist_umf = ((query - query_clamped) / sig_a) ** 2
        
        dist_1 = ((query - mu_1) / sig_a) ** 2
        dist_2 = ((query - mu_2) / sig_a) ** 2
        dist_lmf = torch.max(dist_1, dist_2)
        
        # =========================================================
        # 4. FIRE STRENGTHS & NORMALIZATION
        # =========================================================
        # Sum across the feature dimension (dim=-1)
        # Shape becomes: (Batch, 33, 10)
        
        f_upper = torch.exp(-0.5 * dist_umf)
        f_lower = torch.exp(-0.5 * dist_lmf)

        fire_upper_m = torch.mean(f_upper, dim=1)
        fire_lower_m = torch.mean(f_lower, dim=1)

        fire_softmax_denominator_lower = torch.sum(fire_lower_m, dim=1, keepdim=True) + 1e-4
        fire_softmax_denominator_upper = torch.sum(fire_upper_m, dim=1, keepdim=True) + 1e-4

        fire_upper_mean = fire_upper_m / fire_softmax_denominator_upper
        fire_lower_mean = fire_lower_m / fire_softmax_denominator_lower


        fire_upper = F.softmax(-0.5 * torch.sum(dist_umf, dim=-1), dim=-1)
        fire_lower = F.softmax(-0.5 * torch.sum(dist_lmf, dim=-1), dim=-1)

        c1 = torch.abs(self.c1_consequent).view(1, 1, self.rule_num)  # shape (1, 1, r)
        c2 = torch.abs(self.c2_consequent).view(1, 1, self.rule_num)

        total_fire_upper = fire_upper
        total_fire_lower = fire_lower
       
        fire_mean = (fire_upper_mean + fire_lower_mean) / 2
        total_firing = (total_fire_upper  + total_fire_lower ) / 2  # Avoid division by zero
        self.current_firing_strength = fire_mean
        attention_weight = total_firing

        # =========================================================
        # 6. TSK CONSEQUENT & COMBINATION
        # =========================================================
        # Transform the sequence. Shape: (Batch, 33, 10, 80)
        tsk_sequence = torch.einsum('btf, rcf -> btrc', input_data, self.W_V)
        
        # Multiply the scalar weight against the 80-feature transformation
        weighted_output = attention_weight.unsqueeze(-1) * tsk_sequence
        
        # Sum the 10 rules to get the final temporal output. Shape: (Batch, 33, 80)
        spatial_output = torch.sum(weighted_output, dim=2)
        
        # No permutation needed here! Returns perfectly as (Batch, 33, 80)
        return spatial_output
    
