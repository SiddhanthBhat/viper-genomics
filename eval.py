import torch
from model.viper import VIPER
from utils import compute_metrics

X = torch.randn(200, 101, 4)
y = torch.randint(0, 2, (200,)).numpy()

model = VIPER()
model.load_state_dict(torch.load("viper.pt"))
model.eval()

with torch.no_grad():
    preds = model(X).squeeze().numpy()

print(compute_metrics(y, preds))
