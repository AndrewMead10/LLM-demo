from model import SmallTextGenerator
import torch
import transformers

# x = torch.randint(0, 1000, (1, 100))
# model = SmallTextGenerator()
# y = model(x)

# print(model)
# print(sum([p.numel() for p in model.parameters()]))
# print(y.shape)

model = transformers.OPTModel.from_pretrained("facebook/opt-125m")

print(model)
