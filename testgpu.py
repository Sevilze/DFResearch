import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Get the name of the GPU (if available)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
else:
    print("Running on CPU")
