import torch
import torch.nn as nn
import torchvision


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()

        vgg = torchvision.models.vgg16(pretrained=True)
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg.features[0:9]  # 9,15
        # self.vgg = []
        # self.vgg.append(nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1))
        # self.vgg.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1))
        # for i in range(3):
        #     self.vgg.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1))
        # self.vgg = nn.Sequential(*self.vgg)


    def forward(self, x):
        out = self.vgg(x)
        return out

class RefineNet(nn.Module):
    def __init__(self, net_type:str):
        super(RefineNet, self).__init__()
        self.net_type = net_type

        self.vgg = VGGEncoder()

        self.deconv = []
        self.deconv.append(nn.ConvTranspose2d(128, 32, kernel_size=5, stride=1))
        self.deconv.append(nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1))
        self.deconv.append(nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1))
        self.deconv.append(nn.ConvTranspose2d(32, 2, kernel_size=5, stride=1))
        self.deconv = nn.Sequential(*self.deconv)

        self.fnn = []
        self.fnn.append(nn.Linear(2 + 2 * 32 * 32, 256))
        self.fnn.append(nn.Linear(256, 128))
        self.fnn.append(nn.Linear(128, 2))
        self.fnn = nn.Sequential(*self.fnn)

    def forward(self, x):
        # x = x.expand(3, x.shape[-2], x.shape[-1])
        # x, op = x
        # x = torch.concat([x, op])
        # x = self.vgg(x.float())
        # out = torch.flatten(x, start_dim=0)
        # out = self.fnn(out)
        # out = torch.concat([op, out])
        # out = self.correction(out)

        x, op = x
        x = self.vgg(x)

        mask = self.deconv(x)
        mask = nn.Softmax()(mask)

        out = torch.flatten(mask, start_dim=0)
        out = torch.concat([op, out])
        out = self.fnn(out)

        return out, mask[1:]
