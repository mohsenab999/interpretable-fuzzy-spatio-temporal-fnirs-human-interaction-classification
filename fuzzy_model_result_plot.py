import numpy as np
import matplotlib.pyplot as plt

# 1. Load the SPATIAL firing data
my_data = np.load("spatial_firing.npy")

# Average the batch dimension (axis 0)
# Shape becomes: (8 Rules, 80 Channels)
grid_data = np.mean(my_data, axis=0) 

# 2. Slice the data for Subject 1
# Subject 1 is the first 40 channels. 
# We assume the first 20 are HbO and the next 20 are HbR.
hbo_data = grid_data[:, 0:20]
hbr_data = grid_data[:, 20:40]

# 3. Create a figure with two side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- First Diagram: Subject 1 HbO ---
# No need to transpose (.T) if we want Rules on Y (0-7) and Channels on X (0-19)
im1 = axes[0].imshow(hbo_data, aspect='auto', cmap='plasma')
axes[0].set_title("Subject 1: HbO Spatial Firing Strength")
axes[0].set_xlabel("HbO Channels (0-19)")
axes[0].set_ylabel("Rules (0-7)")
fig.colorbar(im1, ax=axes[0], label="Firing Strength")

# --- Second Diagram: Subject 1 HbR ---
im2 = axes[1].imshow(hbr_data, aspect='auto', cmap='plasma')
axes[1].set_title("Subject 1: HbR Spatial Firing Strength")
axes[1].set_xlabel("HbR Channels (20-39)")
axes[1].set_ylabel("Rules (0-7)") 
fig.colorbar(im2, ax=axes[1], label="Firing Strength")

# Adjust layout and display
plt.tight_layout()
plt.show()