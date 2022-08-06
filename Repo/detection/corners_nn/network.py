import torch
import torch.nn as nn
import torchvision
import numpy as np


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()

        vgg = torchvision.models.vgg16(pretrained=True)
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg_a = vgg.features[0:16]
        self.vgg_b = vgg.features[16:23]
        self.vgg_c = vgg.features[23:-1]

    def forward(self, x):
        preact_a = self.vgg_a(x)
        preact_b = self.vgg_b(preact_a)
        preact_c = self.vgg_c(preact_b)

        return preact_a, preact_b, preact_c


class AttentionMask(nn.Module):
    def __init__(self):
        super(AttentionMask, self).__init__()
        # convolution on 3_3 layer = a
        self.mlif_a = []
        self.mlif_a.append(nn.Conv2d(in_channels=256, out_channels=128,
                                     kernel_size=(1, 1), stride=(1, 1)))
        self.mlif_a.append(nn.Conv2d(in_channels=256, out_channels=128,
                                     kernel_size=(1, 1), stride=(1, 1),
                                     dilation=(2, 2)))
        self.mlif_a.append(nn.Conv2d(in_channels=256, out_channels=128,
                                     kernel_size=(1, 1), stride=(1, 1),
                                     dilation=(2, 2)))
        self.mlif_a.append(nn.Conv2d(in_channels=256, out_channels=128,
                                     kernel_size=(1, 1), stride=(1, 1),
                                     dilation=(3, 3)))

        # convolution on 4_3 layer = b
        self.mlif_b = []
        self.mlif_b.append(nn.Conv2d(in_channels=512, out_channels=128,
                                     kernel_size=(1, 1), stride=(1, 1)))
        self.mlif_b.append(nn.Conv2d(in_channels=512, out_channels=128,
                                     kernel_size=(1, 1), stride=(1, 1),
                                     dilation=(2, 2)))
        self.mlif_b.append(nn.Conv2d(in_channels=512, out_channels=128,
                                     kernel_size=(1, 1), stride=(1, 1),
                                     dilation=(2, 2)))
        self.mlif_b.append(nn.Conv2d(in_channels=512, out_channels=128,
                                     kernel_size=(1, 1), stride=(1, 1),
                                     dilation=(3, 3)))

        # convolution on 5_3 layer = c
        self.mlif_c = []
        self.mlif_c.append(nn.Conv2d(in_channels=512, out_channels=128,
                                     kernel_size=(1, 1), stride=(1, 1)))
        self.mlif_c.append(nn.Conv2d(in_channels=512, out_channels=128,
                                     kernel_size=(1, 1), stride=(1, 1),
                                     dilation=(2, 2)))
        self.mlif_c.append(nn.Conv2d(in_channels=512, out_channels=128,
                                     kernel_size=(1, 1), stride=(1, 1),
                                     dilation=(2, 2)))
        self.mlif_c.append(nn.Conv2d(in_channels=512, out_channels=128,
                                     kernel_size=(1, 1), stride=(1, 1),
                                     dilation=(3, 3)))

        # operations for dimesnsionalyti compliance
        self.dim_a = nn.Conv2d(in_channels=512, out_channels=256,
                               kernel_size=(1, 1), stride=(2, 2))
        self.dim_c = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                        kernel_size=(1, 1), stride=(2, 2))
        self.dim_c.output_padding = (1, 1)
        self.dim_mlif = nn.Conv2d(in_channels=1024, out_channels=2,
                                  kernel_size=(1, 1), stride=(1, 1))

        # dummy variable to register parameters
        self.dummy = nn.Sequential(*self.mlif_c + self.mlif_b + self.mlif_a)

    def forward(self, f_a, f_b, f_c):
        mlif_33 = torch.concat([conv(f_a) for conv in self.mlif_a], dim=1)
        mlif_43 = torch.concat([conv(f_b) for conv in self.mlif_b], dim=1)
        mlif_53 = torch.concat([conv(f_c) for conv in self.mlif_c], dim=1)

        mlif_33 = self.dim_a(mlif_33)
        mlif_53 = self.dim_c(mlif_53)

        mlif = torch.concat([mlif_33, mlif_43, mlif_53], dim=1)
        mlif = self.dim_mlif(mlif)
        att = nn.Softmax(dim=1)(mlif)
        return att


class IntermediateConv(nn.Module):
    def __init__(self,
                 apply_attention_mask=False
                 ):
        super(IntermediateConv, self).__init__()
        # input : 512x8x8
        self.c0 = nn.Conv2d(514, 512, kernel_size=(1, 1), stride=(1, 1))  # 514 or 512
        self.c1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=2)
        self.c2 = nn.Conv2d(514, 512, kernel_size=(1, 1), stride=(1, 1))  # 514 or 512
        self.c3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=2)

    def forward(self, x, attention_mask=None):
        if attention_mask is None:
            x = self.c1(x)
            x = self.c2(x)
        else:
            x = torch.concat([x, attention_mask], dim=1)
            x = self.c0(x)
            x = self.c1(x)
            x = torch.concat([x, attention_mask], dim=1)
            x = self.c2(x)
            x = self.c3(x)
            # x = x * attention_mask[:, 1:, :, :]
            # x = self.c1(x)
            # x = x * attention_mask[:, 1:, :, :]
            # x = self.c2(x)
        return x


class CornerDetector(nn.Module):
    def __init__(self, compute_attention=False):
        super(CornerDetector, self).__init__()
        self.vgg16_enc = VGGEncoder()

        self.intermediate_conv = IntermediateConv()

        self.compute_attention = compute_attention
        if compute_attention: self.attention = AttentionMask()

        self.fnns = []
        self.fnns.append(torch.nn.Linear(512 * 32 * 32, 256))
        self.fnns.append(torch.nn.ReLU())
        self.fnns.append(torch.nn.Linear(256, 32))
        self.fnns.append(torch.nn.ReLU())
        self.fnns.append(torch.nn.Linear(32, 8))
        self.fnns.append(torch.nn.ReLU())
        self.fnns = torch.nn.Sequential(*self.fnns)

    def forward(self, x):
        """
        In this forward call I first apply RCNN for object detection, then accessing last layers
        I train FullyConnectedNN to predict the degree of rotation
        :param x: input batch
        :return: predictions
        """
        a, b, c = self.vgg16_enc(x.float())

        if self.compute_attention:
            att = self.attention(a, b, c)
        else:
            att = None
        final_cnn_enc = self.intermediate_conv(b, att)

        # flatten
        fnn_enc = torch.flatten(final_cnn_enc, 1, 3)
        vgg_out = self.fnns(fnn_enc)

        return vgg_out, att
