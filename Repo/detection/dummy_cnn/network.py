import torch
import torch.nn as nn
import torchvision


class BaseNetDetect(nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 40, kernel_size: int = 11):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyper-parameters"""
        super(BaseNetDetect, self).__init__()

        # I use resnet to extract features from image and use the to predict degree
        # Therefore no_grad, only usage
        self.resnet = torchvision.models.resnet152(pretrained=True)
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters(): p.requires_grad = False

        self.box = []
        self.box.append(torch.nn.Linear(2048, 10**3))
        self.box.append(torch.nn.ReLU())
        self.box.append(torch.nn.Linear(10**3, 10**3))
        self.box.append(torch.nn.ReLU())
        self.box.append(torch.nn.Linear(10**3, 10**2))
        self.box.append(torch.nn.ReLU())
        self.box.append(torch.nn.Linear(10**2, 4))
        self.box.append(torch.nn.ReLU())
        self.box_layers = torch.nn.Sequential(*self.box)

        # self.degree = []
        # self.degree.append(torch.nn.Linear(2048, 10**3))
        # self.degree.append(torch.nn.ReLU())
        # self.degree.append(torch.nn.Linear(10**3, 800))
        # self.degree.append(torch.nn.ReLU())
        # self.degree.append(torch.nn.Linear(800, 500))
        # self.degree.append(torch.nn.ReLU())
        # self.degree.append(torch.nn.Linear(500, 200))
        # self.degree.append(torch.nn.ReLU())
        # self.degree.append(torch.nn.Linear(200, 50))
        # self.degree.append(torch.nn.ReLU())
        # self.degree.append(torch.nn.Linear(50, 1))
        # self.degree.append(torch.nn.ReLU())
        # self.degree_layers = torch.nn.Sequential(*self.degree)

    def forward(self, x):
        x = x/255
        x = x.expand(x.shape[0], 3, x.shape[-2], x.shape[-1]) #making 3 grayscalse channels
        cnn_out = self.resnet(x)
        cnn_out = torch.flatten(torch.flatten(cnn_out,1),1)
        pred_box = self.box_layers(cnn_out)
        # pred_degree = self.degree_layers(cnn_out)
        # pred = torch.concat((pred_box, pred_degree), dim=1)
        return pred_box
