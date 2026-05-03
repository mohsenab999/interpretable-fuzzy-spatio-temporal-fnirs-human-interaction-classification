import numpy as np
import matplotlib.pyplot as plt

# 1. Read the file
my_data = np.load("temporal_firing.npy")

# Average the batch dimension so it's a 2D grid
grid_data = np.mean(my_data, axis=0) 

# 2. Visualize it as a heatmap
plt.imshow(grid_data.T, aspect='auto', cmap='plasma')
plt.colorbar(label="Firing Strength")
plt.xlabel("Channels")
plt.ylabel("Rules")
plt.show()