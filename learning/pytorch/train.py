import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item() * images.size(0)  # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)  # Get the predictions
            total += labels.size(0)  # Total number of labels
            correct += (predicted == labels).sum().item()  # Count correct predictions

        epoch_loss = running_loss / len(train_loader.dataset)  # Average loss
        epoch_accuracy = correct / total  # Calculate accuracy

        # Print results for the current epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')


# Main training script
if __name__ == "__main__":
    # Define transformations for the training dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='data/MNIST', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = SimpleNet()  # Assuming SimpleNet is defined in model.py
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=20)  # Set to 20 epochs or more
