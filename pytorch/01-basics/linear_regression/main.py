# 2차원 좌표에 분포된 데이터를 1차원 직선 방정식을 통해 표현되지 않은
# 데이터를 예측하기 위한 분석 모델

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss asns optimizer
# 손실 함수(cosnt function): 손실 함수란 신경망이 학습할 수 있도록 해주는 지표
# 머신러닝 모델의 출력값과 사용자가 원하는 출력값의 차이, 즉 오차라고 말한다.
# 이 손실 함수 값이 최소화 되도록 하는 가중치와 편향을 찾는것이 학습

# 평균 제곱 오차(Mean Squard Error : MSE)
# 계산이 간편하여 가장 많이 사용되는 손실 함수
# 모델의 출력값과 사용자가 원하는 출력 값 사이의 거리 차리를 오차로 사용
# - 거리 차이를 제곱하면 좋은 점은, 거리 차이가 작은 데이터와 큰 데이터 오차의 차이가
# - 더욱 커진다느 점. 이렇게 되면 어느 부분에서 오차가 두드러지는지 확실히 알 수 있다는 장점.
criterion = nn.MSELoss()

# 데이터의 전부를 보지 않고 조금만 보고 판단 하여 같은 시간에 더 많이 간다.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    #Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Foward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}]. Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')