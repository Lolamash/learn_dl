import numpy as np
import matplotlib.pyplot as plt

# 数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 前向传播
def forward(x, w, b):
    return w * x + b

# 计算 MSE
def mse_loss(w, b):
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val, w, b)
        l_sum += (y_pred - y_val) ** 2
    return l_sum / len(x_data)

# 构造 w 和 b 的网格
w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(-2.0, 2.1, 0.1)

W, B = np.meshgrid(w_range, b_range)

Loss = np.zeros_like(W)

# 计算每个 (w, b) 对应的 loss
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Loss[i, j] = mse_loss(W[i, j], B[i, j])

# 画 3D 曲面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(W, B, Loss)

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE Loss')

plt.show()
