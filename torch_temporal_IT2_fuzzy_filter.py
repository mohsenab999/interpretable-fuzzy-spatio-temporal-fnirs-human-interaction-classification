import torch
import torch.nn as nn
import torch.nn.functional as F

class temporalIT2fuzzyfilter(nn.Module):
    def __init__(self, feature_num=80, time_steps_num=33):
        super(temporalIT2fuzzyfilter, self).__init__()
        self.feature_num = feature_num
        self.time_steps_num = time_steps_num
        self.rule_num = 10

        # --- Antecedent Parameters (Uncertain Mean IT2) ---
        # Sized (10, 33) to cover rules and time steps
        self.matrix_mu_antecedent_first = nn.Parameter(torch.randn(self.rule_num, self.time_steps_num) * 0.1)
        self.matrix_mu_antecedent_second = nn.Parameter(torch.randn(self.rule_num, self.time_steps_num) * 0.1)
        self.matrix_sigma_antecedent = nn.Parameter(torch.ones(self.rule_num, self.time_steps_num))
        
        
        # --- TSK Matrices ---
        # Sized (10, 33, 33) 
        self.W_Q = nn.Parameter(torch.randn(self.rule_num, self.time_steps_num, self.time_steps_num) * 0.1)
        self.W_V = nn.Parameter(torch.randn(self.rule_num, self.time_steps_num, self.time_steps_num) * 0.1)
        self.c1_consequent = nn.Parameter(torch.randn(self.rule_num, 1) * 0.1)
        self.c2_consequent = nn.Parameter(torch.randn(self.rule_num, 1) * 0.1)

    def forward(self, input_data):
        # input_data shape: (Batch, 80, 33)
        
        # =========================================================
        # 1. ANTECEDENT QUERY (W_Q * X)
        # =========================================================
        # Shape: (Batch, Features, Rules, Time) -> (Batch, 80, 10, 33)
        query = torch.einsum('bft, rot -> bfro', input_data, self.W_Q)
        
        # =========================================================
        # 2. RESHAPE PARAMETERS FOR BROADCASTING
        # =========================================================
        mu_1 = self.matrix_mu_antecedent_first.view(1, 1, self.rule_num, self.time_steps_num)
        mu_2 = self.matrix_mu_antecedent_second.view(1, 1, self.rule_num, self.time_steps_num)
        # Force sigma to be strictly positive to prevent NaN division
        sig_a = torch.abs(self.matrix_sigma_antecedent).view(1, 1, self.rule_num, self.time_steps_num) + 1e-4
        
        # Find the mathematical upper and lower boundaries of the uncertain mean
        mu_lower = torch.min(mu_1, mu_2)
        mu_upper = torch.max(mu_1, mu_2)
        
        # =========================================================
        # 3. CALCULATE IT2 DISTANCES (Piecewise logic replacement)
        # =========================================================
        # UMF Distance Trick: Clamp the query between the two means.
        # If the query is between them, distance becomes 0.0!
        query_clamped = torch.clamp(query, min=mu_lower, max=mu_upper)
        dist_umf = ((query - query_clamped) / sig_a) ** 2
        
        # LMF Distance Trick: Take the maximum distance to either mean.
        dist_1 = ((query - mu_1) / sig_a) ** 2
        dist_2 = ((query - mu_2) / sig_a) ** 2
        dist_lmf = torch.max(dist_1, dist_2)
        
        # =========================================================
        # 4. FIRE STRENGTHS & NORMALIZATION
        # =========================================================
        # Sum across the time dimension (dim=-1) to apply Product T-Norm
        # Shape becomes: (Batch, 80, 10)
        
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
        # Transform the sequence. Shape: (Batch, 80, 10, 33)
        tsk_sequence = torch.einsum('bft, rot -> bfro', input_data, self.W_V)
        
        # Multiply the scalar weight against the 33-step sequence
        weighted_output = attention_weight.unsqueeze(-1) * tsk_sequence
        
        # Sum the 10 rules to get the final spatial output. Shape: (Batch, 80, 33)
        temporal_output = torch.sum(weighted_output, dim=2)
        
        # =========================================================
        # 7. PERMUTE FOR THE TEMPORAL LAYER
        # =========================================================
        # Output Shape becomes: (Batch, 33, 80)
        return temporal_output
