import torch
import torch.nn as nn

from torchsummary import summary

# Define a CNN model for 513x337 spectrogram classification
class SonoModel(nn.Module):
    def __init__(self, num_classes=5):  # Adjust num_classes based on your dataset
        super(SonoModel, self).__init__()
        
        # Define layers
        self.classifier = nn.Sequential(
            # Convolutional layers: Conv -> ReLU -> MaxPool
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2),   # (1,513,337) -> (8,513,337)
            nn.Dropout(0.3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # (8,513,337) -> (8,256,168)

            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=0),  # (8,256,168) -> (32,252,164)
            nn.BatchNorm2d(32),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # (32,252,164) -> (32,126,82)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0), # (32,126,82) -> (64,122,78)
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # (64,122,78) -> (64,61,39)

            # Fully connected layer for classification
            nn.Flatten(),                                                                   # (152256)
            nn.Linear(152256, 64),
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
model = SonoModel().to(DEVICE)
# summary(model, (513, 337))
