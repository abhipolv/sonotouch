import torch
from sonoModel import SonoModel
from sonoModelLive import SonoModelLive
from sonoDataset import SonoDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import seaborn as sns
import time
import numpy as np
import os

# Test function with confusion matrix (percentage) & model saving
def test_model():
    model.to(DEVICE)

    # Batch loop
    predicted_labels = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.float().to(DEVICE), labels.long().to(DEVICE)  # move to GPU

            # Forward pass
            out = model(inputs)
        
            # Update accuracy counters
            _, predictions = torch.max(out, 1)
            predicted_labels = np.append(predicted_labels, predictions.cpu().numpy())  # move to CPU before appending
            true_labels = np.append(true_labels, labels.cpu().numpy())

    # Calculate accuracy & confusion matrix
    test_acc = 100 * (predicted_labels == true_labels).sum().item() / len(predicted_labels)
    print("Test accuracy: ", test_acc, "%", sep="")

    cm = confusion_matrix(true_labels, predicted_labels, labels=[0,1,2,3,4], normalize='true')
    sns.heatmap(cm, annot=True, fmt=".1%", cmap='Blues',
                xticklabels=["0","1","2","3","4"], yticklabels=["0","1","2","3","4"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("CNN Confusion Matrix")


# Function to evaluate single sample inference time on test dataset
def test_single_sample_inference(model, test_loader):
    # Move model to CPU for timing
    model.to("cpu")

    # Iterate through each sample in DL
    t_sample = []

    with torch.no_grad():
        # Batch loop
        for inputs, labels in test_loader:
            t_start = time.time()

            # Single sample inference
            out = model(inputs.float())
            _, prediction = torch.max(out, 1)

            t_end = time.time()
            t_sample.append((t_end-t_start) / len(inputs))  # avg
    
    t_avg = sum(t_sample) / len(t_sample)
    print("Average single-sample inference time (CPU):", t_avg, "seconds")


# Test the model
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(DEVICE)

model = SonoModel().to(DEVICE)
# model = SonoModelLive().to(DEVICE)
model.load_state_dict(torch.load("sono_model.pt", weights_only=False), strict=False)

model.eval()

# test_dl = torch.load("sonotemp_dl.pth", weights_only=False)
test_path = os.getcwd() + "/data/alyssa_04212025T1820_44100hz/"  # SonoModel - contains neutral class
dataset = SonoDataset(test_path)
test_dl = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

test_single_sample_inference(model, test_dl)  # separate test, since model is run on GPU
test_model()
plt.show()

# save model dict & load to check
# torch.save(model.state_dict(), "abhi_model.pt")
# model2 = SonoModelLive().to(DEVICE)
# model2.load_state_dict(torch.load("sono_dict.pt", weights_only=False), strict=False)