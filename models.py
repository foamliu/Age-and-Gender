import torch.nn.functional as F
import torchvision
from torch import nn
from torchsummary import summary

from config import *


class AgeGenPredModel(nn.Module):
    def __init__(self):
        super(AgeGenPredModel, self).__init__()

        resnet = torchvision.models.resnet18(pretrained=True)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(512, 512)
        self.age_pred = nn.Linear(512, age_num_classes)

        self.fc2 = nn.Linear(512, 512)
        self.gen_pred = nn.Linear(512, gen_num_classes)

    def forward(self, images):
        x = self.resnet(images)
        last_conv_out = x.view(-1, 512)  # (batch_size, 512)

        age_out = F.relu(self.fc1(last_conv_out))
        age_out = self.age_pred(age_out)

        gen_out = F.relu(self.fc2(last_conv_out))
        gen_out = self.gen_pred(gen_out)

        return age_out, gen_out


if __name__ == "__main__":
    model = AgeGenPredModel().to(device)
    summary(model, (3, 224, 224))
