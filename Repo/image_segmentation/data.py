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


class BillSet(Dataset):
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
        self.masks = sorted(glob.glob(os.path.join(image_dir, "**", "*.npy"), recursive=True))
        self.masks = np.array(self.masks)[indexes]
        self.masks = np.repeat(self.masks, repeats=coefficient)
        self.seed = seed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im, msk = self.images[idx], self.masks[idx]
        im, msk = Image.open(im).convert('L'), Image.fromarray(np.load(msk))
        im, msk = self.preprocess_image(im, msk)
        return im, msk

    def preprocess_image(self, image: Image, mask: Image) -> (Image, Image):
        """This method received one image and mask, adds noise and padding
            @:param image: Image, generated picture of bill
            @:param mask: Image, shows the pixels of interest
            @:return tuple: with edited image and mask"""

        # seed
        if not self.seed is None:
            rnd.seed(self.seed)
        # random resize
        percentage = rnd.randint(0, 51) / 100
        inc_or_dec = -1#1 if rnd.normal() >= .4 else -1
        image.thumbnail(
            size=(int(image.width * (1 + percentage * inc_or_dec)), int(image.height * (1 + percentage * inc_or_dec))),
            resample=Image.ANTIALIAS)
        mask.thumbnail(
            size=(int(mask.width * (1 + percentage * inc_or_dec)), int(mask.height * (1 + percentage * inc_or_dec))),
            resample=Image.ANTIALIAS)

        # define background size and colour, should be bigger than image
        rnd_width = rnd.randint(image.width, image.width * 3)
        rnd_height = rnd.randint(image.height, image.height * 3)
        background = (rnd.randint(0, 256),)

        # define padding from left and top
        width_difference = abs(rnd_width - image.width)+1
        left_pad = rnd.randint(0, width_difference)

        height_difference = abs(rnd_height - image.height)+1
        top_pad = rnd.randint(0, height_difference)

        # doing padding
        new_im = Image.new(image.mode, (rnd_width, rnd_height), background)
        new_im.paste(image, (left_pad, top_pad))
        new_mask = Image.new(mask.mode, (rnd_width, rnd_height), (0,))
        new_mask.paste(mask, (left_pad, top_pad))

        # add rotation
        if rnd.normal() >= .4:
            degrees = rnd.randint(0, 20)
            inc_or_dec = 1 if rnd.normal() >= .5 else -1
        else:
            degrees = 0
            inc_or_dec = 0
        new_im = new_im.rotate(degrees * inc_or_dec, fillcolor=background)
        new_mask = new_mask.rotate(degrees * inc_or_dec, fillcolor=(0,))

        # adding noise
        arr_nim = np.array(new_im).astype(np.float64)
        noise_percentage = rnd.randint(1, 30) / 100
        noise_matrix = rnd.randint(0, 100, size=arr_nim.shape) / 100  # noise which I will apply
        criterion_matrix = rnd.normal(0.5, 0.2, size=arr_nim.shape)  # probability that pixel will be changed
        noise_mask = (criterion_matrix <= noise_percentage).astype(np.uint8)  # masking which pixels will change
        noise_matrix *= noise_mask  # zeroing pixels which don't change
        noise_matrix += (1 - noise_mask)  # bring zeros to ones
        arr_nim *= noise_matrix  # apply noise

        return arr_nim.astype(np.uint8), np.array(new_mask).astype(np.uint8)


class BoxedBillSet(BillSet):
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

        #boxes are desribed by x0, y0, x1, y1
        #in my case I have 2 boxes per each image
        boxes = np.zeros(shape=(2,4))
        ones = np.argwhere(msk==100)
        boxes[0] = np.array([ones[0,0], ones[0,1], ones[-1,0], ones[-1,1]])
        twos = np.argwhere(msk==200)
        boxes[1] = np.array([twos[0,0], twos[0,1], twos[-1,0], twos[-1,1]])

        #areas -- there are 2 numbers corresponding to b-boxes areas
        areas = np.array([ones.shape[0], twos.shape[0]])

        #modified masks -- masks are splitted to 2 different layers
        modified_masks = np.zeros(shape=(2, msk.shape[0], msk.shape[1]))
        modified_masks[0][ones] = 1
        modified_masks[1][twos] = 1

        #creating final target dictionary
        target = {
                  'boxes': torch.from_numpy(boxes.astype(np.int64)),
                  'area': torch.from_numpy(areas),
                  'labels': torch.from_numpy(np.array([1, 2]).astype(np.int64)),
                  'image_id': torch.from_numpy(np.array([idx]).astype(np.int64)),
                  'masks': list(modified_masks),  # this is changed in collate fn!!!
                  'iscrowd': torch.from_numpy(np.array([False, False]).astype(np.uint8))
                  }

        return im, target
