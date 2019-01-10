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
        self.fc1 = nn.Linear(2048, 2048)
        self.age_cls_pred = nn.Linear(2048, age_cls_unit)

        self.fc2 = nn.Linear(2048, 2048)
        self.gen_cls_pred = nn.Linear(2048, 2)

    def forward(self, images):
        x = self.resnet(images)
        last_conv_out = x.view(-1, 2048)  # (batch_size, 2048)

        age_pred = F.relu(self.fc1(last_conv_out))
        age_pred = F.softmax(self.age_cls_pred(age_pred), 1)

        gen_pred = F.relu(self.fc2(last_conv_out))
        gen_pred = self.gen_cls_pred(gen_pred)

        return age_pred, gen_pred


if __name__ == "__main__":
    model = AGModel().to(device)
    summary(model, (3, 224, 224))
