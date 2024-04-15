import torch
import matplotlib.pyplot as plt


device = torch.device('cpu')

N, Din, Dout = 64, 4, 2

x = torch.randn(N, Din, device=device)
y = torch.randn(N, Dout, device=device)

model = torch.nn.Sequential(
    torch.nn.Linear(Din, Dout),
)

lr = 1e-6

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_arr = []

for t in range(50000):
    # 5-1. 예측 값을 계산
    y_hat = model(x)

    # 5-2. loss 계산
    # loss = 0.5*(y_hat - y).pow(2).sum()
    # loss = torch.nn.functional.mse_loss(y_hat, y, reduction='sum') * 0.5
    loss = torch.nn.functional.mse_loss(y_hat, y)
    loss_arr.append(loss)

    # 5-3. 경사하강법 적용
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

plt.plot([loss.cpu().detach() for loss in loss_arr])
