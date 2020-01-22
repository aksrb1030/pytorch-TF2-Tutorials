import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# ================================================= #
#            1. Basic autograd example 1            #
# ================================================= #

# Create tensors
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build as computational graph
y = w * x + b

# Compute gradients
y.backward()

# Print out the gradients

# print (x.grad)
# print (w.grad)
# print (b.grad)

# ================================================= #
#            2. Basic autograd example 2            #
# ================================================= #

# Create tensors of shape (10, 3) and (10, 2).
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Build a fully connected layer
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# Build loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr = 0.01)

#Forward pass
pred = linear(x)

# Compute loss
loss = criterion(pred, y)
print('loss', loss.item())

# Backward pass
loss.backward()

# Print out the gradients
print ('dl/dw: ', linear.weight.grad)
print ('dl/db: ', linear.bias.grad)

# 1-step gradient descent
optimizer.step()

pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())
