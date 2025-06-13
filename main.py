import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Define the Lion optimizer (since it's not included in PyTorch by default)
class Lion(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update weights
                update = exp_avg.sign()
                p.add_(update, alpha=-group["lr"])

        return loss


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms - Reduce image size to decrease memory usage
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Reduced from 512x512 to 224x224
        transforms.ToTensor(),
    ]
)

# Dataset paths
train_path = "data/train"
val_path = "data/val"

# Datasets
train_data = datasets.ImageFolder(root=train_path, transform=transform)
val_data = datasets.ImageFolder(root=val_path, transform=transform)

# Dataloaders - Reduce batch size to decrease memory usage
batch_size = 16  # Reduced from 32 to 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# Define custom model with customizable CNN layers
class CustomCNNModel(nn.Module):
    def __init__(self, num_cnn_layers=3, dropout_rate=0.2):
        super(CustomCNNModel, self).__init__()
        
        # Start with fewer filters and gradually increase
        cnn_layers = []
        
        # First CNN layer
        cnn_layers.append(nn.Conv2d(3, 16, kernel_size=3, padding=1))
        cnn_layers.append(nn.BatchNorm2d(16))
        cnn_layers.append(nn.ReLU())
        cnn_layers.append(nn.MaxPool2d(2))
        
        # Variable number of additional CNN layers
        previous_channels = 16
        for i in range(num_cnn_layers - 1):
            # Double the number of channels in each layer
            out_channels = previous_channels * 2
            
            cnn_layers.append(nn.Conv2d(previous_channels, out_channels, kernel_size=3, padding=1))
            cnn_layers.append(nn.BatchNorm2d(out_channels))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d(2))
            
            previous_channels = out_channels
        
        self.cnn_block = nn.Sequential(*cnn_layers)
        
        # Calculate the size of the flattened features
        # For 224x224 input with num_cnn_layers MaxPool2d operations
        final_size = 224 // (2 ** num_cnn_layers)
        flattened_size = previous_channels * final_size * final_size
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 2)  # 2 classes: cat and dog
        )
    
    def forward(self, x):
        x = self.cnn_block(x)
        x = self.classifier(x)
        return x

# Create model with specified number of CNN layers
num_cnn_layers = 5  # You can change this number to add more CNN layers
model = CustomCNNModel(num_cnn_layers=num_cnn_layers, dropout_rate=0.2)  # Reduced dropout rate
model = model.to(device)

# Print model structure
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# Using Lion optimizer with reduced learning rate
optimizer = Lion(model.parameters(), lr=5e-5, weight_decay=1e-6)  # Reduced values

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
num_epochs = 50
best_val_acc = 0

print(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():  # Important for memory usage
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    val_accuracies.append(val_acc)
    
    # Print progress
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
    
    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/cat_dog_custom_cnn_best.pth")
        print(f" New best model saved at epoch {epoch + 1} with validation accuracy: {val_acc:.2f}%")

# Save final model
torch.save(model.state_dict(), "models/cat_dog_custom_cnn_final.pth")
print(" Final model saved!")