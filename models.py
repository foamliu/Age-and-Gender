import torch.nn.functional as F
import torchvision
from torch import nn
from torchsummary import summary

from config import *


class AGModel(nn.Module):
    def __init__(self):
        super(AGModel, self).__init__()

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, num_classes)

    def forward(self, images):
        x = self.resnet(images)
        x = x.view(-1, 2048)  # (batch_size, 2048)
        x = self.fc1(x)
        print('x.size(): ' + str(x.size()))
        return F.softmax(x, dim=1)


if __name__ == "__main__":
    model = AGModel().to(device)
    summary(model, (3, 256, 256))
