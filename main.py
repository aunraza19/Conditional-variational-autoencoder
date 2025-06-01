import torch
from cvae import CVAE
from dataloader import get_dataloaders
from train import train_model
from test import test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loader, test_loader = get_dataloaders()

train_model(model, train_loader, optimizer, device, num_epochs=10)
test_model(model, test_loader, device)
