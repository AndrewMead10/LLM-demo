from model import SmallTextGenerator
import torch

x = torch.randint(0, 1000, (1, 100))
model = SmallTextGenerator()
y = model(x)
print(model)
print(sum([p.numel() for p in model.parameters()]))
print(y.shape)
