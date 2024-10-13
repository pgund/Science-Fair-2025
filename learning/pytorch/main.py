import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleNet
from train import train_model
from utils import save_model, load_model

# Set up data transformations and loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Train the model
model = train_model(train_loader, epochs=5)

# Save the trained model
save_model(model, 'model_weights.pth')

# Optionally, load and test the model
loaded_model = load_model(SimpleNet(), 'model_weights.pth')
