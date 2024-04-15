import torch
import matplotlib.pyplot as plt


device = torch.device('cpu')

N, Din, Dout = 64, 4, 2

x = torch.randn(N, Din, device=device)
y = torch.randn(N, Dout, device=device)
w = torch.randn(Din, Dout, device=device, requires_grad=True)

lr = 1e-6
loss_arr = []

for t in range(50000):
    # 5-1. 예측 값을 계산
    y_hat = x.mm(w)

    # 5-2. loss 계산
    loss = (0.5 * (y_hat - y) ** 2).sum()
    loss_arr.append(loss)

    # 5-3. 경사하강법 적용
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_()


plt.plot([loss.detach() for loss in loss_arr])
