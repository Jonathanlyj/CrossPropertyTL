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

set_seed(42)

# Simple one-layer regression model
class SimpleRegressor(nn.Module):
    def __init__(self):
        super(SimpleRegressor, self).__init__()
        self.linear = nn.Linear(10, 1)  # One fully connected layer

    def forward(self, x):
        return self.linear(x)

# Generate random input and target data
input_data = torch.randn(100, 10)  # 100 samples, 10 features
target_data = torch.randn(100, 1)  # 100 target values

# Initialize model, loss function, and optimizer
model = SimpleRegressor()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop (just one epoch)
model.train()
for epoch in range(1):
    optimizer.zero_grad()  # Zero gradients
    outputs = model(input_data)  # Forward pass
    loss = criterion(outputs, target_data)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

# Print the loss after one epoch
print(f"Loss after one epoch: {loss.item()}")
