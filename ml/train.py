import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sonoDataset import SonoDataset
from sonoModel import SonoModel
from sonoModelLive import SonoModelLive
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import os

# Ensure GPU/CPU compatibility
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(DEVICE)

# Load dataset
train_path = os.getcwd() + "/data/combined_user/"  # SonoModel - contains neutral class
dataset = SonoDataset(train_path)

# Split dataset: 70% train, 15% val, 15% test
train_data, test_data, val_data = random_split(dataset, [0.70, 0.15, 0.15])

# Create DataLoaders
batch_size = 8
train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)

# -- Sanity check: view random Spectrogram
random_idx = random.randint(0, len(train_data))
random_sample_image = torch.squeeze(train_data[random_idx][0]) # Squeeze removes the extra leading dimension
random_sample_label = train_data[random_idx][1]
print(random_sample_image.size())

fig, ax = plt.subplots()
img = ax.imshow(random_sample_image.real, extent=[0, 1, 0, 44100/2], origin='lower', aspect='auto', cmap='jet')
# img.set_clim(vmin=-100, vmax=20)
plt.title(f"Image Label: {random_sample_label}")
# plt.show()

# Initialize model, loss function, and optimizer
model = SonoModel(num_classes=5).to(DEVICE)
# model = SonoModelLive(num_classes=4).to(DEVICE)

learning_rate = 1e-4
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()


# Training function
def train_model(is_plot=False):
    num_epoch = 20
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # Epoch loop:
    for epoch in range(num_epoch):
        epoch_loss = []
        num_correct = 0
        num_total = 0
        t_start = time.time()

        # Batch loop:
        for inputs, labels in train_dl:
            inputs, labels = inputs.float().to(DEVICE), labels.long().to(DEVICE)  # move to GPU
            
            # Forward pass
            optimizer.zero_grad()  # zero optimizer
            out = model(inputs)
            loss = loss_function(out, labels)
            epoch_loss.append(loss.item())

            # Backwards pass
            loss.backward()
            optimizer.step()

            # Update accuracy counters
            _, predicted_labels = torch.max(out, 1)
            num_total += labels.size(0)
            num_correct += (predicted_labels == labels).sum().item()

        # Save epoch loss & accuracy
        train_loss.append(sum(epoch_loss) / len(epoch_loss))
        train_acc.append(100 * num_correct/num_total)

        # Validate model
        loss, acc = validate_model(model, val_dl)
        val_loss.append(loss)
        val_acc.append(acc)
    
        # Print epoch stats
        t_end = time.time()

        print("Epoch:", epoch)
        print("\tAvg. training loss:", train_loss[-1])
        print("\tTraining accuracy: ", train_acc[-1], "%", sep="")
        print("\tAvg. validation loss:", val_loss[-1])
        print("\tValidation accuracy: ", val_acc[-1], "%", sep="")
        print("\tEpoch time:", t_end-t_start, "seconds")
    
    # Plot validation / training accuracy & loss
    if is_plot:
        t = np.arange(num_epoch)
        plt.figure()

        plt.subplot(2,1,1)
        plt.plot(t,train_acc, label="Training Accuracy")
        plt.plot(t,val_acc, label="Validation Accuracy")
        plt.ylabel("Accuracy")

        plt.subplot(2,1,2)
        plt.plot(t,train_loss, label="Training Loss")
        plt.plot(t,val_loss, label="Validation Los")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.show()


# Validation function
def validate_model(model, val_loader):
    model.eval()

    # Epoch loop:
    with torch.no_grad():
        epoch_loss = []
        num_correct = 0
        num_total = 0

        for inputs, labels in val_loader:
            inputs, labels = inputs.float().to(DEVICE), labels.long().to(DEVICE)  # move to GPU

            # Forward pass
            out = model(inputs)
            loss = loss_function(out, labels)
            epoch_loss.append(loss.item())
           
           # Update accuracy counters
            _, predicted_labels = torch.max(out, 1)
            num_total += labels.size(0)
            num_correct += (predicted_labels == labels).sum().item()

        # Return validation loss & accuracy
        val_loss = sum(epoch_loss) / len(epoch_loss)
        val_acc = 100 * num_correct/num_total
        return val_loss, val_acc


# Train the model
train_model(is_plot=True)
torch.save(model.state_dict(), "sonotemp_dict.pt")  # save dictionary
torch.save(test_dl, "sonotemp_dl.pth")  # save testing data
