import torch
import torch.nn as nn

from torchsummary import summary

# Define a CNN model for 513x130 spectrogram classification
# Input based on peak segmentation (0.4s), with no neutral class
class SonoModelLive(nn.Module):
    def __init__(self, num_classes=4):  # Adjust num_classes based on your dataset
        super(SonoModelLive, self).__init__()
        
        # Define layers
        self.classifier = nn.Sequential(
            # Convolutional layers: Conv -> ReLU -> MaxPool
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2),   # (1,513,130) -> (8,513,130)
            # nn.Dropout(0.3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # (8,513,337) -> (8,256,65)

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0),  # (8,256,65) -> (16,252,61)
            nn.BatchNorm2d(16),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # (16,252,61) -> (16,126,30)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0), # (16,126,30) -> (32,122,26)
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # (32,122,26) -> (32,61,13)

            # Fully connected layer for classification
            nn.Flatten(),                                                                   # (25376)
            nn.Linear(25376, 64),
            # nn.Dropout(0.2),
            nn.Linear(64, num_classes),

            # Softmax layer for class decision
            # nn.Softmax()  # only if loss function is NOT CrossEntropy
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # add dim for channels
        return self.classifier(x)


# Sanity check: checking model dims
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = SonoModelLive().to(DEVICE)
summary(model, (513, 130))