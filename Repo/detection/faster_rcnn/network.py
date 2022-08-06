import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class FastRCNN(nn.Module):
    def __init__(self):
        """Fast RCNN with masking, tuning on resnet 50 trained on COCO set"""
        super(FastRCNN, self).__init__()

        self.rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        num_classes = 2
        in_features = self.rcnn.roi_heads.box_predictor.cls_score.in_features
        self.rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    def forward(self, x):
        """
        In this forward call I first apply RCNN for object detection, then accessing last layers
        I train FullyConnectedNN to predict the degree of rotation
        :param x: input batch
        :return: predictions
        """
        im, trg, optimize = x
        im = [el/255 for el in im]
        im = [el.expand(3, el.shape[-2], el.shape[-1]) for el in im]

        if optimize: # I switched to using inbuilt training function, because my implementation was faulty
            self.rcnn.train()
            rcnn_loss = self.rcnn(im,trg)
        else: rcnn_loss = {"dummy":0}

        self.rcnn.eval()
        rcnn_pred = self.rcnn(im)
        return rcnn_pred, rcnn_loss
