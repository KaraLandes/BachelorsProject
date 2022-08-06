import torch
import torch.nn as nn
import torchvision


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()

        vgg = torchvision.models.vgg16(pretrained=True)
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg.features[0:23]#15

    def forward(self, x):
        out = self.vgg(x)
        return out

class RefineNet(nn.Module):
    def __init__(self, net_type:str):
        super(RefineNet, self).__init__()
        self.net_type = net_type

        self.vgg = VGGEncoder()

        self.conv = []
        self.conv.append(nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.conv.append(nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.conv = nn.Sequential(*self.conv)

        self.fnn = []
        self.fnn.append(nn.Linear(32 * 8 * 8, 32))
        self.fnn.append(nn.Linear(32, 2))
        self.fnn = nn.Sequential(*self.fnn)

    def forward(self, x):
        """
        In this forward call I first apply RCNN for object detection, then accessing last layers
        I train FullyConnectedNN to predict the degree of rotation
        :param x: input batch
        :return: predictions
        """

        x = x.expand(3, x.shape[-2], x.shape[-1])
        x = self.vgg(x.float())
        x = self.conv(x)
        out = torch.flatten(x, start_dim=0)
        out = self.fnn(out)


        return out
