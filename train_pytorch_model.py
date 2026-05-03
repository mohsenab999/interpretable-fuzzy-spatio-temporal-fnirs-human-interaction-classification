import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from your_extraction_script import extract_fnirs_trials_class
from IFT2_Spatio_Temporal_BCI_Network import IT2SpatioTemporalBCINetwork
from center_recovery import recover_fuzzy_centers

def load_raw_tensor_data(subj1_path, subj2_path):
    tensor_sub1 = extract_fnirs_trials_class.extract_fnirs_trials(subj1_path)
    tensor_sub2 = extract_fnirs_trials_class.extract_fnirs_trials(subj2_path)
    dyad_tensor = np.concatenate((tensor_sub1, tensor_sub2), axis=1)
    return dyad_tensor

# --- 2. LOOP THROUGH ALL 20 PAIRS ---

# List the folder names of your 20 pairs. 
# (Update this list to match the exact names of the folders in your dataset!)
pair_folders = [
    '1003', '1004', '1005', '1006', '1007', '1008',
    '1009', '1010', '1011', '1012', '1013', '1014',
    '1015', '1016', '1017', '1018', '1019', '1020',
    '1021', '1022' , '1023', '1024', '1025', '1026',
    '1027', '3004', '3005', '3006', '3007', '3008', 
    '3009', '3010', '3011', '3012', '3013', '3014', 
    '3015', '3016', '3017', '3018', '3019', '3020', 
    '3021', '3022', '3023', '3024', '3025', '3026',
]

# Create empty lists to hold the data for all pairs
all_class_0_data = []
all_class_1_data = []

print("--- Starting Bulk Data Extraction ---")

for pair in pair_folders:
    print(f"\n>>> Extracting Data for Pair: {pair} <<<")
    
    # Define the dynamic paths for this specific pair
    block1_sub1 = f'dataset/fnirs_raw_data/{pair}/block1/Subject1/'
    block1_sub2 = f'dataset/fnirs_raw_data/{pair}/block1/Subject2/'
    
    block2_sub1 = f'dataset/fnirs_raw_data/{pair}/block2/Subject1/'
    block2_sub2 = f'dataset/fnirs_raw_data/{pair}/block2/Subject2/'
    
    try:
        # Load Block 1 (No Touch)
        pair_class_0 = load_raw_tensor_data(block1_sub1, block1_sub2)
        all_class_0_data.append(pair_class_0)
        
        # Load Block 2 (Hand Holding)
        pair_class_1 = load_raw_tensor_data(block2_sub1, block2_sub2)
        all_class_1_data.append(pair_class_1)
        
    except Exception as e:
        # If one pair has a missing file or error, print it but keep going!
        print(f"Skipping pair {pair} due to error: {e}")

# --- 3. STACK ALL DATA TOGETHER ---
print("\n--- Stacking the Full Dataset ---")

# Stack the lists into giant numpy arrays
X_class_0 = np.vstack(all_class_0_data)
X_class_1 = np.vstack(all_class_1_data)

# Create the labels (0s and 1s) based on the total number of trials gathered
y_class_0 = np.zeros(X_class_0.shape[0])
y_class_1 = np.ones(X_class_1.shape[0])

# Combine Class 0 and Class 1
X_numpy = np.vstack((X_class_0, X_class_1))
y_numpy = np.concatenate((y_class_0, y_class_1))

print(f"Total Combined Dataset Shape: {X_numpy.shape}")

# Split into 80% Training Data and 20% Testing Data
X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy, test_size=0.2, random_state=42)

# ... (The rest of your PyTorch tensor conversion and Training Loop stays EXACTLY the same!) ...

# Add this new import to the top of your script!
from torch.utils.data import TensorDataset, DataLoader

# --- 3. CONVERT TO PYTORCH TENSORS ---
# (Keep this part exactly the same)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# --- NEW: CREATE THE DATALOADER ---
# Package the tensors into a Dataset, then put them in a DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# batch_size=32 is perfectly safe for memory. 
# shuffle=True mixes the data so the model doesn't memorize the order!
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# --- 4. THE PYTORCH TRAINING LOOP ---
print("\n--- Starting Neural Network Training ---")

model = IT2SpatioTemporalBCINetwork().to(device)
criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0007) 

num_epochs = 400

for epoch in range(num_epochs):
    
    # ==========================
    #      TRAINING PHASE
    # ==========================
    model.train() # 1. Set model to training mode (activates your Dropout layer)
    train_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)

    # ==========================
    #      PRINT RESULTS
    # ==========================
    # Print every 10 epochs to keep the terminal clean
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f}")

        
# --- 5. TEST THE TRAINED MODEL ---
print("\n--- Testing the Network ---")
model.eval() # Turn off training mode

with torch.no_grad(): # Don't track calculus anymore to save memory
    test_predictions = model(X_test_tensor)
    
    # Convert probability (e.g. 0.8) to class (1.0)
    test_predictions_class = test_predictions.round() 
    
    # Calculate Accuracy
    correct = (test_predictions_class == y_test_tensor).sum().item()
    total = y_test_tensor.size(0)
    accuracy = (correct / total) * 100

print(f"Final PyTorch Prediction Accuracy: {accuracy:.2f}%")

# ==========================================
# 5. POST-TRAINING ANALYSIS
# ==========================================
print("--- Extracting IT2 Fuzzy Rules ---")
import torch
import numpy as np

model.eval() 

with torch.no_grad(): # Ensures we don't accidentally track gradients
    # ---------------------------------------------------------
    # SPATIAL LAYER EXTRACTION
    # ---------------------------------------------------------
    # 1. Extract the upper and lower means, then average them to find the "center"
    spatial_mu_1 = model.spatial_layer.matrix_mu_antecedent_first.detach().cpu()
    spatial_mu_2 = model.spatial_layer.matrix_mu_antecedent_second.detach().cpu()
    spatial_centers = (spatial_mu_1 + spatial_mu_2) / 2.0
    
    # 2. Extract the projection tensor (W_Q) and the firing strength
    spatial_W_Q = model.spatial_layer.W_Q.detach().cpu()
    spatial_firing = model.spatial_layer.current_firing_strength.detach().cpu()
    
    # ---------------------------------------------------------
    # TEMPORAL LAYER EXTRACTION
    # ---------------------------------------------------------
    temporal_mu_1 = model.temporal_layer.matrix_mu_antecedent_first.detach().cpu()
    temporal_mu_2 = model.temporal_layer.matrix_mu_antecedent_second.detach().cpu()
    temporal_centers = (temporal_mu_1 + temporal_mu_2) / 2.0
    
    temporal_W_Q = model.temporal_layer.W_Q.detach().cpu()
    temporal_firing = model.temporal_layer.current_firing_strength.detach().cpu()

# ---------------------------------------------------------
# SAVE TO NUMPY FILES
# ---------------------------------------------------------
np.save("spatial_centers_it2.npy", spatial_centers.numpy())
np.save("spatial_W_Q.npy", spatial_W_Q.numpy())
np.save("spatial_firing.npy", spatial_firing.numpy())

np.save("temporal_centers_it2.npy", temporal_centers.numpy())
np.save("temporal_W_Q.npy", temporal_W_Q.numpy())
np.save("temporal_firing.npy", temporal_firing.numpy())

print("IT2 Extraction and saving complete!")