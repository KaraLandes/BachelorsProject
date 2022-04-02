import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class BaseNetDetect(nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 3, kernel_size: int = 11):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyper-parameters"""
        super(BaseNetDetect, self).__init__()

        self.cnn = []
        for i in range(n_hidden_layers):
            self.cnn.append(torch.nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels,
                                            kernel_size=kernel_size, bias=True, padding=int(kernel_size / 2)))
            # self.cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*self.cnn)
        self.preoutput_layer = torch.nn.Conv2d(in_channels=n_kernels, out_channels=2,
                                               kernel_size=kernel_size, bias=True,
                                               padding=int(kernel_size / 2))
        self.output_layer = torch.nn.ReLU()

    def forward(self, x):
        x = torch.tensor(x, dtype=self.cnn[0].weight.dtype)
        x = x.reshape(x.size()[0], 1, x.size()[1], x.size()[2])
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = (self.preoutput_layer(cnn_out))  # apply output layer (N, n_kernels, X, Y) -> (N, 2, X, Y)
        return pred


class MRCNN():
    def __init__(self, num_classes: int):
        """
        Initailisation class allows use pretrained pytorch model
        :param num_classes: number of classes one want to mask, 2 in our case: logo and content
        """
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256  # OOM maybe lower?
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

        self.model = model

    def get_model(self):
        return self.model
