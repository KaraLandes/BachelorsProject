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

def collate_fn(batch):
    max_r, max_c = 0, 0
    for im, msk in batch:
        if im.shape[0] > max_r: max_r = im.shape[0]
        if im.shape[1] > max_c: max_c = im.shape[1]
    target_shape = (max_r, max_c)

    # shape = elements in batch, rows, cols
    collated_ims = np.zeros(shape=(len(batch), target_shape[0], target_shape[1]))
    collated_msk = np.zeros(shape=(len(batch), target_shape[0], target_shape[1]))

    # fill with values
    for i, (im, msk) in enumerate(batch):
        collated_ims[i][:im.shape[0], :im.shape[1]] = im
        collated_msk[i][:msk.shape[0], :msk.shape[1]] = msk

    # normalise
    # TODO ???

    # tensors
    collated_msk = torch.from_numpy(collated_msk.astype(np.float32))
    collated_ims = torch.from_numpy(collated_ims.astype(np.float32))

    return collated_ims, collated_msk