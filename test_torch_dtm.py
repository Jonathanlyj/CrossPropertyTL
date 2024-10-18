import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(42)

# Check if a GPU is available, otherwise use the CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

device = torch.device("cpu")


# Custom model with dense, dropout, and batch normalization layers
class CustomRegressor(nn.Module):
    def __init__(self):
        super(CustomRegressor, self).__init__()
        self.fc1 = nn.Linear(10, 64)  # Dense layer (10 input features, 64 output)
        self.bn1 = nn.BatchNorm1d(64)  # Batch Normalization layer
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout rate
        self.fc2 = nn.Linear(64, 1)  # Final Dense layer (64 input features, 1 output)
    
    def forward(self, x):
        x = self.fc1(x)  # Apply first fully connected layer
        x = self.bn1(x)  # Apply batch normalization
        x = torch.relu(x)  # Apply ReLU activation function
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Apply final fully connected layer
        return x

# Generate random input and target data
input_data = torch.randn(100, 10).to(device)  # 100 samples, 10 features, move to device
target_data = torch.randn(100, 1).to(device)  # 100 target values, move to device

# Initialize the custom model, loss function, and optimizer
model = CustomRegressor().to(device)  # Move model to device
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop (just one epoch)
model.train()
for epoch in range(1000):
    optimizer.zero_grad()  # Zero gradients
    outputs = model(input_data)  # Forward pass
    loss = criterion(outputs, target_data)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

# Print the loss after one epoch
print(f"Loss after training: {loss.item()}")