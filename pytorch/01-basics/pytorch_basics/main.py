import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# autograd 패키지는 Tensor로 수행한 모든 연산에 대하여 자동 미분 기능을 제공
# 모든 계산을 마친 후에 .backward()를 호출하면, 자동으로 모든 기울기가 계산

# ================================================= #
#            1. Basic autograd example 1            #
# ================================================= #

# Create tensors
# .reqires_grad 속성을 true로 설정하면 그 tenser에서 이뤄진 모든 연산들을
# 추적 하기 시작한다. 계산이 완료된 후 .backward()를 호출하여 모든 gradient를 자동으로 계산
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

print (x)
print (w)
print (b)

# Build as computational graph
y = w * x + b

# Compute gradients
y.backward()

# Print out the gradients

print (x.grad)
print (w.grad)
print (b.grad)

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
# MSE 평균 제곱 오차 loss
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

# # ================================================= #
# #            3. Loading data from numpy             #
# # ================================================= #

# # Create a numpy array.
# x = np.array([[1,2], [3, 4]])

# #Convert the numpy array to a torch tensor.
# y = torch.from_numpy(x)

# #Convert the numpy array to a numpy array.
# z = y.numpy()

# # ================================================= #
# #                  4. Input pipline                 #
# # ================================================= #

# # Download and construct CIFAR - 10 dataset.
# train_dataset = torchvision.datasets.CIFAR10(root='../../data/', 
# train=True, 
# transform=transforms.ToTensor(),
# download=False)

# # Fetch one data pair (read data from disk)
# image, label = train_dataset[0]
# print (image.size())
# print (label)

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
# batch_size=64,
# shuffle=True)

# # When interation starts, queue and thread start to load data from files.
# data_iter = iter(train_loader)

# # Mini-batch images and labels.
# images, labels = data_iter.next()

# # Actual usage of the data loader is as below
# for images, labels in train_loader:
#     # Training code should be written here.
#     pass

# # ================================================= #
# #        5. Input pipline for custom dataset        #
# # ================================================= #

# # You should build your custom dataset as below

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         # TODO
#         # 1. Initialize file paths or a list of file names.
#         pass

# def __getitem__(self, index):
#     # TODO
#     # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open)
#     # 2. Preprocess the data (e.g. torchvision.Transform)
#     # 3. Return a data pair (e.g. image and label)
#     pass
# def __len__(self):
#     # You should change 0 to the total size of your dataset.
#     return 0

# # You can the use the prebuild data loader.
# custom_dataset = CustomDataset()
# train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=64, shuffle=True)

# # ================================================= #
# #                 6. Pretrained model               #
# # ================================================= #

# # Download and load the pretrained ResNet-18
# resnet = torchvision.models.resnet18(pretrained=True)

# # If you want the finetune only the top layer of model, set as below.
# for param in resnet.parameters():
#     param.requires_grad = False

# # Replace the top layer for finetuning
# resnet.fc = nn.Linear(resnet.fc.in_features, 100)

# # Foward pass
# images = torch.randn(64, 3, 244, 244)
# outputs = resnet(images)
# print (outputs.size())

# # ================================================= #
# #            7. Save and load the model             #
# # ================================================= #

# # Save and load the entire model.
# torch.save(resnet, 'model.ckpt')
# model = torch.load('model.ckpt')

# # Save and load only the model parameters (recommended)
# torch.save(resnet.state_dict(), 'params.ckpt')
# resnet.load_state_dict(torch.load('params.ckpt'))
