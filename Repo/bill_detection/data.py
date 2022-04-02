import os
import glob
import numpy as np
import torch
from numpy import random as rnd
import warnings

warnings.filterwarnings("ignore")
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset


class BillOnBackGroundSet(Dataset):
    """
    Generated Bills dataset.
    """

    def __init__(self, image_dir: str, indexes: list, coefficient=1, seed=None):
        """
        Initialsiation.
        @:param images_dir: String path to directory where generated bills are held
        @:param indexes: List of images ids to be assigned to this set
        @:param coefficient: Number of times by which to boost ub set size
                             (since random transformations are applied here)
        """
        self.images = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))
        self.images = np.array(self.images)[indexes]
        self.images = np.repeat(self.images, repeats=coefficient)

        self.rnd = rnd
        self.rnd.seed(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        trg = self.images[idx]
        trg = Image.open(trg).convert('L')
        im = self.preprocess_image(trg)

        return im, trg

    def preprocess_image(self, image: Image) -> (Image, float):
        """This method received one image and mask, adds noise and padding
            @:param image: Image, generated picture of bill
            @:param mask: Image, shows the pixels of interest
            @:return tuple: with edited image and rotation angle"""

        # random resize
        percentage = self.rnd.randint(5, 41) / 100  # making smaller helps to fit everything to CUDA
        inc_or_dec = -1  # 1 if self.rnd.normal() >= .4 else -1
        image.thumbnail(
            size=(int(image.width * (1 + percentage * inc_or_dec)), int(image.height * (1 + percentage * inc_or_dec))),
            resample=Image.ANTIALIAS)

        # define background size and colour, should be bigger than image
        rnd_width = self.rnd.randint(image.width, image.width * 2)
        rnd_height = self.rnd.randint(image.height, image.height * 2)
        background = (self.rnd.randint(0, 256),)

        # define padding from left and top
        width_difference = abs(rnd_width - image.width) + 1
        left_pad = self.rnd.randint(0, width_difference)

        height_difference = abs(rnd_height - image.height) + 1
        top_pad = self.rnd.randint(0, height_difference)

        # doing padding
        new_im = Image.new(image.mode, (rnd_width, rnd_height), background)
        new_im.paste(image, (left_pad, top_pad))

        # add rotation
        if self.rnd.normal() >= 0:
            degrees = self.rnd.randint(0, 20)
            inc_or_dec = 1 if self.rnd.normal() >= .5 else -1
        else:
            degrees = 0
            inc_or_dec = 0
        new_im = new_im.rotate(degrees * inc_or_dec, fillcolor=background)

        # adding noise
        arr_nim = np.array(new_im).astype(np.float64)
        noise_percentage = self.rnd.randint(1, 30) / 100
        noise_matrix = self.rnd.randint(0, 100, size=arr_nim.shape) / 100  # noise which I will apply
        criterion_matrix = self.rnd.normal(0.5, 0.2, size=arr_nim.shape)  # probability that pixel will be changed
        noise_mask = (criterion_matrix <= noise_percentage).astype(np.uint8)  # masking which pixels will change
        noise_matrix *= noise_mask  # zeroing pixels which don't change
        noise_matrix += (1 - noise_mask)  # bring zeros to ones
        arr_nim *= noise_matrix  # apply noise

        return arr_nim.astype(np.uint8), degrees


class BoxedBillSet(BillOnBackGroundSet):
    """
    This class is created as extension of BillSet class.
    Additional functionality is reflected in getitem method, where
    additionally b-boxed are returned
    """
    pass

    def __getitem__(self, idx):
        im, msk = self.images[idx], self.masks[idx]
        im, msk = Image.open(im).convert('L'), Image.fromarray(np.load(msk))
        im, msk = self.preprocess_image(im, msk)

        # boxes are described by x0, y0, x1, y1
        # in my case I have 2 boxes per each image
        boxes = np.zeros(shape=(2, 4))
        ones = np.argwhere(msk == 100)
        x0, x1 = np.min(ones[:, 1]), np.max(ones[:, 1])
        y0, y1 = np.min(ones[:, 0]), np.max(ones[:, 0])
        boxes[0] = np.array([x0, y0, x1, y1])
        twos = np.argwhere(msk == 200)
        x0, x1 = np.min(twos[:, 1]), np.max(twos[:, 1])
        y0, y1 = np.min(twos[:, 0]), np.max(twos[:, 0])
        boxes[1] = np.array([x0, y0, x1, y1])

        # areas -- there are 2 numbers corresponding to b-boxes areas
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # modified masks -- masks are splitted to 2 different layers
        modified_masks = [np.zeros(shape=(msk.shape[0], msk.shape[1])), np.zeros(shape=(msk.shape[0], msk.shape[1]))]
        modified_masks[0] = np.where(msk == 100, 1, 0).astype(np.uint8)
        modified_masks[1] = np.where(msk == 200, 1, 0).astype(np.uint8)

        # creating final target dictionary
        target = {
            'boxes': torch.from_numpy(boxes.astype(np.float32)),
            'area': torch.from_numpy(areas),
            'labels': torch.from_numpy(np.array([1, 2]).astype(np.int64)),
            'image_id': torch.from_numpy(np.array([idx]).astype(np.int64)),
            'masks': modified_masks,  # this is changed in collate fn!!!
            'iscrowd': torch.from_numpy(np.array([False, False]).astype(np.int64))
        }

        return im, target
