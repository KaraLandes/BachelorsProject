import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from ..data import AbstractBillOnBackGroundSet, AbstractRealBillSet


class FRCNNBillOnBackGroundSet(AbstractBillOnBackGroundSet):
    def form_target(self, image: Image, seed: int) -> tuple:
        output_shape = self.output_shape
        image, mask = self.preprocess_image(image, seed, output_shape)

        mask = np.array(mask)
        mask[mask < 200] = 0
        mask[mask >= 200] = 256
        rows = np.argwhere(np.isclose(mask, 256))[:, 0]
        columns = np.argwhere(np.isclose(mask, 256))[:, 1]
        boxes = []  # target array collects all transformations done to image in order to revert t afterwards
        boxes.append(min(rows))
        boxes.append(min(columns))
        boxes.append(max(rows))
        boxes.append(max(columns))
        if boxes[2] - boxes[0] <= 0: boxes[2] = 80
        if boxes[3] - boxes[1] <= 0: boxes[3] = 80
        area = (max(rows) - min(rows)) * (max(columns) - min(columns))
        target = {}
        target["boxes"] = torch.Tensor([boxes])
        target["labels"] = torch.Tensor([1]).to(torch.int64)
        target["masks"] = torch.Tensor([mask]).to(torch.int64)
        target["image_id"] = torch.Tensor([seed]).to(torch.int64)
        target["area"] = torch.Tensor([area]).to(torch.int64)
        target["iscrowd"] = torch.Tensor([0]).to(torch.int64)

        image = np.array(image)
        image = np.array([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
        return image, target


class FRCNNRealBillSet(AbstractRealBillSet):
    def form_target(self, image: Image, mask: Image, seed: int, im_name: str):
        output_shape = self.output_shape
        image, mask = self.preprocess_image(image, mask, output_shape, im_name=im_name)

        mask = np.array(mask)
        mask[mask < 200] = 0
        mask[mask >= 200] = 255
        rows = np.argwhere(mask == 0)[:, 0]
        columns = np.argwhere(mask == 0)[:, 1]
        boxes = []  # target array collects all transformations done to image in order to revert t afterwards
        boxes.append(min(rows))
        boxes.append(min(columns))
        boxes.append(max(rows))
        boxes.append(max(columns))

        if boxes[2] - boxes[0] <= 0:
            boxes[2] = 80
        if boxes[3] - boxes[1] <= 0:
            boxes[3] = 80
        area = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])

        target = {}
        target["boxes"] = torch.Tensor([boxes])
        target["labels"] = torch.Tensor([1]).to(torch.int64)
        target["masks"] = torch.Tensor([mask]).to(torch.int64)
        target["image_id"] = torch.Tensor([seed]).to(torch.int64)
        target["area"] = torch.Tensor([area]).to(torch.int64)
        target["iscrowd"] = torch.Tensor([0]).to(torch.int64)

        image = np.array(image)
        image = np.array([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
        return image, target
