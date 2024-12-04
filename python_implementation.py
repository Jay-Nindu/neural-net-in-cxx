import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

data = fetch_california_housing()
print(data.feature_names)

X, y = data.data, data.target


#train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

#data standardisation
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

myData = {
    "train": torch.utils.data.TensorDataset(X_train, y_train),
    "test": torch.utils.data.TensorDataset(X_test, y_test),
}

myLoader = {x: DataLoader(myData[x], batch_size=1) for x in ["train", "test"]}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 24),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(24, 12),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(12, 6),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(6, 1)
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output

def train_model(model, criterion, optimizer, num_epochs=25):
    # Create a temporary directory to save training checkpoints
    hist = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in myLoader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        
                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                optimizer.step() # batch grad descent (after going through all data)

            epoch_loss = running_loss / 50
            if phase == 'test':
                hist.append(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f}')

        print()
    return hist


model_ft = NeuralNetwork()
model_ft = model_ft.to(device)


criterion = nn.MSELoss()


#Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.1)

#Decay LR by a factor of 0.1 every 7 epochs
hist = train_model(model_ft, criterion, optimizer_ft,
                       num_epochs=50)

plt.plot(hist)
plt.show()
