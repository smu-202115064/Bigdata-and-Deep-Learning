import torch
import matplotlib.pyplot as plt


device = torch.device('cpu')


class MyOneLayerNet(torch.nn.Module):
    def __init__(self, Din, Dout) -> None:
        super(MyOneLayerNet, self).__init__()
        self.linear = torch.nn.Linear(Din, Dout)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat


N, Din, Dout = 64, 4, 2

x = torch.randn(N, Din, device=device)
y = torch.randn(N, Dout, device=device)

model = MyOneLayerNet(Din, Dout)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

loss_arr = []

for t in range(50000):
    # 5-1. 예측 값을 계산
    y_hat = model(x)

    # 5-2. loss 계산
    loss = torch.nn.functional.mse_loss(y_hat, y)
    loss_arr.append(loss)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

plt.plot([loss.cpu().detach() for loss in loss_arr])
