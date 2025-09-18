import torch
print("cuda_available:", torch.cuda.is_available())
print("torch.cuda:", torch.version.cuda)
print("torch version:", torch.__version__)
