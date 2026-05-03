import torch
import torch.nn as nn
import torch.linalg as linalg

def recover_fuzzy_centers(query_centers: torch.Tensor, projection_layer: nn.Linear) -> torch.Tensor:
    """
    Recovers the fuzzy centers from the query embedding space back to the 
    original input space (e.g., raw EEG/fNIRS channels or time points).
    
    Args:
        query_centers: The learned fuzzy centers in the query space. 
                       Expected shape: (num_rules, out_features)
        projection_layer: The nn.Linear layer used to project inputs to queries (W^Q).
        
    Returns:
        recovered_centers: The centers mapped back to the raw input space.
                           Shape: (num_rules, in_features)
    """
    with torch.no_grad(): # No need to track gradients for visualization
        # 1. Extract weights (W) and bias (b)
        W = projection_layer.weight.data  # Shape: (out_features, in_features)
        
        if projection_layer.bias is not None:
            b = projection_layer.bias.data # Shape: (out_features)
        else:
            b = torch.zeros(W.size(0), device=W.device)
            
        # 2. Compute the Moore-Penrose pseudoinverse of W (W^+)
        # Under the hood, this computes the SVD: W = U * \Sigma * V^T
        W_pinv = linalg.pinv(W) # Shape: (in_features, out_features)
        
        # 3. Recover the original input: x = W^+ (y - b)
        y_minus_b = query_centers - b
        
        # In PyTorch, batch operations are row-based.
        # To match the math x = W^+ * (y - b), we use matrix multiplication:
        recovered_centers = torch.matmul(y_minus_b, W_pinv.T)
        
        return recovered_centers

# --- Example Usage ---
# Assuming you have 10 rules, input dimension of 64 (channels), and query dimension of 32
num_rules = 10
in_features = 64
out_features = 32

# Mock layer and learned centers
query_projection = nn.Linear(in_features, out_features)
learned_query_centers = torch.randn(num_rules, out_features)

# Recover!
raw_centers = recover_fuzzy_centers(learned_query_centers, query_projection)
print(f"Recovered shape: {raw_centers.shape}") # Expected: [10, 64]