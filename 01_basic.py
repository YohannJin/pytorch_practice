from __future__ import print_function
import torch

# a tensor without initialization
x = torch.empty(5, 3)
print(x)

# generate a random 5*3 tensor
x = torch.rand(5, 3)
print(x)

# a all 0 torch
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# directly with data
x = torch.tensor([5.5, 3])
print(x)

# initialize from a existing one like tensor
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)

# get the dimensionality of tensor
print(x.size())

# addition
# 2 ways
y = torch.rand(5, 3)
print(x+y)
print(torch.add(x+y))

# addition with reference
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# adds in place
y.add_(x)
print(y)

# use index
print(x[:, 1])


# change tensor size
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# read the value from tensor: use .item()
x = torch.randn(1)
print(x)
print(x.item())











