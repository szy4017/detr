import torch
import os
from thop import profile

from util.misc import nested_tensor_from_tensor_list

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
device = torch.device('cuda')
model = torch.load('../checkpoints/State-DETR.pth')
input = torch.rand(3, 540, 960)
x = nested_tensor_from_tensor_list([input])
x = x.to(device)
model.to(device)
flops, params = profile(model, inputs=(x,))
print(flops)
print(params)

total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))