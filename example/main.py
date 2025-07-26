import torch
from transformer.model import GPTModel, GPT_CONFIG_124M


torch.manual_seed(42)


model = GPTModel(GPT_CONFIG_124M)

