import numpy as np
import matplotlib.pyplot as plt


N, Din, Dout = 64, 4, 2

x = np.random.randn(N, Din)
y = np.random.randn(N, Dout)
w = np.random.randn(Din, Dout)

lr = 1e-6
loss_arr = []

for t in range(50000):
    # 5-1. 예측 값을 계산
    y_hat = np.matmul(x, w)

    # 5-2. loss 계산
    loss = (0.5 * (y_hat - y) ** 2).sum()
    loss_arr.append(loss)

    # 5-3. 경사하강법 적용
    grad_y_hat = y_hat - y
    grad_w = np.matmul(x.T, grad_y_hat)

    # 5-4. w를 갱신
    w -= lr * grad_w


plt.plot(loss_arr)
