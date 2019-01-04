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
        # self.embedding = nn.Linear(feature_size, embedding_size)
        # self.sigmoid = nn.Sigmoid()
        self.fine_tune()

    def forward(self, images):
        out = self.resnet(images)
        out = out.view(-1, feature_size)  # (batch_size, 2048)
        # out = self.embedding(out)
        # out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    model = AGModel().to(device)
    summary(model, (3, 256, 256))
