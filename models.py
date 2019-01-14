import torch.nn.functional as F
import torchvision
from torch import nn
from torchsummary import summary

from config import *


class AgeGenPredModel(nn.Module):
    def __init__(self):
        super(AgeGenPredModel, self).__init__()

        resnet = torchvision.models.resnet50(pretrained=True)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(6144, 512)
        self.age_pred = nn.Linear(512, age_num_classes)

        self.fc2 = nn.Linear(6144, 512)
        self.gen_pred = nn.Linear(512, gen_num_classes)

    def forward(self, images):
        x = self.resnet(images)  # [N, 512, 4, 3]
        # last_conv_out = x.view(-1, 6144)  # [N, 512]
        #
        # age_out = F.relu(self.fc1(last_conv_out))  # [N, 512]
        # age_out = F.softmax(self.age_pred(age_out), dim=1)  # [N, 101]
        #
        # gen_out = F.relu(self.fc2(last_conv_out))  # [N, 512]
        # gen_out = F.softmax(self.gen_pred(gen_out), dim=1)  # [N, 2]

        return None, None # age_out, gen_out


if __name__ == "__main__":
    model = AgeGenPredModel().to(device)
    summary(model, (3, 224, 224))
