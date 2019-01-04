import torchvision
from torch import nn

from config import *
from torchsummary import summary


class AGModel(nn.Module):
    """
    Encoder.
    """

    def __init__(self, embedding_size=123):
        super(AGModel, self).__init__()

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # self.embedding = nn.Linear(feature_size, embedding_size)
        # self.sigmoid = nn.Sigmoid()
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)
        out = out.view(-1, feature_size)  # (batch_size, 2048)
        # out = self.embedding(out)
        # out = self.sigmoid(out)
        return out

    def fine_tune(self):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False


if __name__ == "__main__":
    model = AGModel(embedding_size=123).to(device)
    summary(model, (3, 224, 224))
