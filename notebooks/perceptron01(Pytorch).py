import torch
import matplotlib.pyplot as plt


device = torch.device('cpu')

N, Din, Dout = 64, 4, 2

x = torch.randn(N, Din, device=device)
y = torch.randn(N, Dout, device=device)
w = torch.randn(Din, Dout, device=device)

lr = 1e-6
loss_arr = []

for t in range(50000):
    # 5-1. 예측 값을 계산
    y_hat = x.mm(w)

    # 5-2. loss 계산
    loss = (0.5 * (y_hat - y) ** 2).sum()
    loss_arr.append(loss)

    # 5-3. 경사하강법 적용
    grad_y_hat = y_hat - y
    grad_w = x.T.mm(grad_y_hat)

    # 5-4. w를 갱신
    w -= lr * grad_w


plt.plot(loss_arr)
