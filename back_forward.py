import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt

x_data = torch.tensor([1.0, 2.0, 3.0])
y_data = torch.tensor([2.0, 4.0, 6.0])

w = torch.tensor([1.0], requires_grad=True)

def forward(x):
    return w * x

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

w_history = []
loss_history = []

print("predict (before training)", 4, forward(torch.tensor([4.0])).item())

for epoch in range(30):
    for x, y in zip(x_data, y_data):

        l = loss(x, y)
        l.backward()

        w_history.append(w.item())
        loss_history.append(l.item())

        with torch.no_grad():
            w -= 0.01 * w.grad

        w.grad.zero_()

print("predict (after training)", 4, forward(torch.tensor([4.0])).item())

plt.plot(w_history, loss_history)
plt.xlabel("w")
plt.ylabel("loss")
plt.grid()
plt.show()