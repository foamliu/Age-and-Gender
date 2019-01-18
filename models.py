import torch.nn.functional as F
import torchvision
from torch import nn
from torchsummary import summary
from torchvision import transforms

from config import *

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class AgeGenPredModelClassification(nn.Module):
    def __init__(self):
        super(AgeGenPredModelClassification, self).__init__()

        resnet = torchvision.models.resnet18(pretrained=True)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(512, 512)
        self.age_pred = nn.Linear(512, age_num_classes)

        self.fc2 = nn.Linear(512, 512)
        self.gen_pred = nn.Linear(512, gen_num_classes)

    def forward(self, images):
        x = self.resnet(images)  # [N, 512, 1, 1]
        last_conv_out = x.view(-1, 512)  # [N, 512]

        age_out = F.relu(self.fc1(last_conv_out))  # [N, 512]
        age_out = F.softmax(self.age_pred(age_out), dim=1)  # [N, 101]

        gen_out = F.relu(self.fc2(last_conv_out))  # [N, 512]
        gen_out = F.softmax(self.gen_pred(gen_out), dim=1)  # [N, 2]

        return age_out, gen_out


class AgeGenPredModel(nn.Module):
    def __init__(self):
        super(AgeGenPredModel, self).__init__()

        resnet = torchvision.models.resnet18(pretrained=True)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.pool = nn.AvgPool2d(4)
        # self.fc1 = nn.Linear(512, 512)
        self.age_pred = nn.Linear(512, 1)

        # self.fc2 = nn.Linear(512, 512)
        self.gen_pred = nn.Linear(512, gen_num_classes)

        nn.init.xavier_uniform(self.age_pred.weight)
        nn.init.xavier_uniform(self.gen_pred.weight)

    def forward(self, images):
        x = self.resnet(images)  # [N, 512, 1, 1]
        x = self.pool(x)
        x = x.view(-1, 512)  # [N, 512]

        # age_out = F.relu(self.fc1(x))  # [N, 512]
        age_out = self.age_pred(x)  # [N, 1]

        # gen_out = F.relu(self.fc2(x))  # [N, 512]
        gen_out = F.softmax(self.gen_pred(x), dim=1)  # [N, 2]

        return age_out, gen_out


if __name__ == "__main__":
    model = AgeGenPredModel().to(device)
    summary(model, (3, 112, 112))
