import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from model.viper import VIPER

cfg = yaml.safe_load(open("configs/config.yaml"))

X = torch.randn(1000, 101, 4)
y = torch.randint(0, 2, (1000, 1)).float()

ds = TensorDataset(X, y)
dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

model = VIPER(cfg["model"]["channels"])
opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
loss_fn = torch.nn.BCELoss()

for epoch in range(cfg["train"]["epochs"]):
    for xb, yb in dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch} | Loss {loss.item():.4f}")

torch.save(model.state_dict(), "viper.pt")
