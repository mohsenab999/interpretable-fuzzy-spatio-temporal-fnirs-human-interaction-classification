import torch
import gc

# 1. Delete the model or large tensors if you are completely done with them
# del model, inputs, outputs

# 2. Run Python's garbage collector to remove unreferenced objects
gc.collect()

# 3. Empty the PyTorch CUDA cache
torch.cuda.empty_cache()