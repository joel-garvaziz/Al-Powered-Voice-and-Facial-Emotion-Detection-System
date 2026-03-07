import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())