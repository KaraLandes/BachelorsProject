import os
import glob

import numpy as np
from numpy import random as rnd
import warnings

warnings.filterwarnings("ignore")
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset

import cv2


class AbstractBillOnBackGroundSet(Dataset):
    """
    Generated Bills dataset.
    """

    def __init__(self, image_dir: str, coefficient=1, seed=None, output_shape: tuple = (3000, 3000)):
        """
        Initialsiation.
        @:param images_dir: String path to directory where generated bills are held
        @:param indexes: List of images ids to be assigned to this set
        @:param coefficient: Number of times by which to boost ub set size
                             (since random transformations are applied here)
        """
        self.images = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))
        self.imdir = image_dir
        self.images = np.array(self.images)#[:8]
        self.images = np.repeat(self.images, repeats=coefficient)
        self.output_shape = output_shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        im = Image.open(im).convert('RGB')
        im, trg = self.form_target(im, idx)

        return im, trg

    def form_target(self, image: Image, seed: int) -> tuple:
        """
        This method runs unified preprocessing procedure
        and then forms target w.r.t. the network structure
        :param image: Image of the bill
        :param seed: seeding random generator
        :param output_shape: shape of final image and mask for inputs unification
        :return: tuple (array of image, specific target)
        """
        raise NotImplementedError

    def preprocess_image(self, image: Image, seed: int, output_shape=(256, 256)):
        """
        Method preprocesses bill image on fly. Given bill generated image, we place it on some background texture,
        rotate, add noise, generate corresponding mask
        :param image: bill Image
        :param seed: random seed
        :param output_shape: shape of output image, extended as black frame (for CNN)
        :return: tuple (bill on background as array, target as tuple of b-box coordinates and degree of rotation)
        """

        # 1 create mask as white rectangle, which will be changed together with bill
        mask = np.ones(shape=(image.height, image.width))
        mask.fill(256)
        mask = Image.fromarray(mask)

        # 2 rotate bill and mask
        np.random.seed(seed)
        degrees = np.random.randint(0, 180)
        if degrees % 90 < 3: degrees + 5  # patch
        image = image.rotate(degrees, fillcolor=(0,), expand=True)
        mask = mask.rotate(degrees, fillcolor=(0,), expand=True)

        # 3 create my backround as terxture image
        textures_dir = os.path.abspath(os.path.join(self.imdir, os.pardir, os.pardir))
        textures_dir = os.path.join(textures_dir,
                                    "materials_for_preprocessing",
                                    "textures")
        textures = sorted(glob.glob(os.path.join(textures_dir, "**", "*.jpg"), recursive=True))
        background = np.random.choice(textures, size=1)[0]
        rnd_width = np.random.randint(image.width, image.width * 2)
        rnd_height = np.random.randint(image.height, image.height * 2)
        background = Image.open(background).convert('RGB')
        background = background.transform(size=(rnd_width, rnd_height),
                                          method=Image.EXTENT,
                                          data=(0, 0, background.width, background.height))  # resize, enlarge
        # 4 everything to arrays
        background = np.array(background)
        image = np.array(image)
        mask = np.array(mask)

        # 5 where mask is white (256) we insert bill to texture
        width_difference = abs(rnd_width - image.shape[1]) + 1
        left_pad = np.random.randint(0, width_difference)

        height_difference = abs(rnd_height - image.shape[0]) + 1
        top_pad = np.random.randint(0, height_difference)

        subarray_where_bill_will_be = background[top_pad:top_pad + image.shape[0],
                                      left_pad:left_pad + image.shape[1]]
        where_mask_is_white = mask != 0
        subarray_where_bill_will_be[where_mask_is_white] = image[where_mask_is_white]
        # insert this back to texture
        background[top_pad:top_pad + image.shape[0], left_pad:left_pad + image.shape[1]] = subarray_where_bill_will_be

        # 6 add some noise
        background = background.astype(np.float64)
        noise_percentage = np.random.randint(1, 20) / 100
        noise_matrix = np.random.randint(0, 100, size=background.shape) / 100  # noise which I will apply
        criterion_matrix = np.random.normal(0.5, 0.2, size=background.shape)  # probability that pixel will be changed
        noise_mask = (criterion_matrix <= noise_percentage).astype(np.uint8)  # masking which pixels will change
        noise_matrix *= noise_mask  # zeroing pixels which don't change
        noise_matrix += (1 - noise_mask)  # bring zeros to ones
        background *= noise_matrix  # apply noise

        # 7 do the mask as big as background
        mask_big = np.zeros(shape=background.shape[:-1])
        mask_big[top_pad:top_pad + image.shape[0], left_pad:left_pad + image.shape[1]] = mask
        mask = mask_big

        # 8 unifying size
        image = Image.fromarray(background.astype(np.uint8))
        mask = Image.fromarray(mask)
        dims = output_shape
        image.thumbnail(size=dims, resample=Image.ANTIALIAS)
        mask.thumbnail(size=dims, resample=Image.ANTIALIAS)
        new_im = Image.new(image.mode, output_shape, (0,))
        new_im.paste(image, (0, 0))

        mask_b = Image.new(mask.mode, output_shape, (0,))
        mask_b.paste(mask, (0, 0))

        return new_im, mask_b

    def calculate_corners(self, mask):
        corners = cv2.goodFeaturesToTrack(mask, 4, 0.01, 5)
        corners = np.squeeze(corners)
        if len(corners) != 4:
            np.append(corners, [0, 0])
            np.append(corners, [0, 0])

        corners_ordered = []
        temp = [c[0] + c[1] for c in corners]
        corners_ordered.append(np.argmin(temp))

        while len(corners_ordered) != 4:
            min_angle, idx = np.inf, 0
            oy = np.array([0, 1]) / np.linalg.norm([0, 1])
            for j in range(len(corners)):
                if np.isin(j, corners_ordered):
                    pass
                else:
                    c = corners[j]
                    first_corner = corners[corners_ordered[0]]
                    c_origin = np.array((c[0] - first_corner[0], c[1] - first_corner[1]))
                    unit_c = c_origin / np.linalg.norm(c_origin)
                    dot_product = np.dot(oy, unit_c)
                    angle = np.arccos(dot_product)
                    if angle < min_angle:
                        min_angle = angle
                        idx = j

            corners_ordered.append(idx)
        corners = [corners[j] for j in corners_ordered]
        return corners


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


class AbstractRealBillSet(Dataset):
    def __init__(self, image_dir: str, output_shape: tuple = (3000, 3000), coefficient=2):
        """
        Initialsiation.
        @:param images_dir: String path to directory where bills and masks are held
        :type output_shape: object
        """
        self.images = sorted(glob.glob(os.path.join(image_dir, "*.jpg"), recursive=False))
        self.images = self.images*coefficient
        self.masks = sorted(glob.glob(os.path.join(image_dir, "[m]*.png"), recursive=False))
        self.masks = self.masks*coefficient
        self.output_shape = output_shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        msk = self.masks[idx]
        im = (Image.open(im).convert('RGB'))
        msk = (Image.open(msk).convert('L'))
        im, trg = self.form_target(im, msk, idx, im_name=self.images[idx])

        return im, trg

    def form_target(self, image: Image, mask: Image, seed: int, im_name: str):
        raise NotImplementedError

    def preprocess_image(self, image: Image, mask: Image, output_shape=(256, 256), im_name=None, seed=None) -> (
    np.array, list):
        """This method received one image and mask, adds noise and padding
            @:param image: Image, generated picture of bill
            @:param output_shape: tuple specifying collation field shape
            @:return tuple: with image, coordinates of bbox and degree of rotation"""

        dims = output_shape
        image.thumbnail(size=dims, resample=Image.ANTIALIAS)
        mask.thumbnail(size=dims, resample=Image.ANTIALIAS)

        rotate_neg = ["real_spar_2_007", "real_spar_2_008", "real_spar_2_009", "real_spar_2_010",
                      "real_spar_2_011", "real_spar_2_012", "real_spar_2_013", "real_spar_2_014",
                      "real_spar_2_036", "real_spar_2_037", "real_spar_2_038", "real_spar_2_039",
                      "real_spar_2_040", "real_spar_2_041", "real_spar_2_043", "real_spar_2_044",
                      "real_spar_2_045", "real_billa_007", "real_spar_3_038", "real_spar_3_039",
                      "real_spar_3_040", "real_spar_3_041", "real_spar_3_042", "real_spar_4_002",
                      "real_spar_4_003", "real_spar_4_004"]

        im_name = im_name.split("/")[-1][:-4]
        if mask.width != image.width and np.isin(im_name, rotate_neg):
            mask = mask.rotate(-90, expand=True)
        elif mask.width != image.width:
            mask = mask.rotate(90, expand=True)
        elif np.isin(im_name, rotate_neg):
            mask = mask.rotate(180, expand=True)

        # some random rotation
        if not seed is None: np.random.seed(seed)
        rnd_n = np.random.random()
        if rnd_n < .25:
            image = image.rotate(90, expand=True)
            mask = mask.rotate(90, expand=True)
        elif rnd_n < .5:
            image = image.rotate(180, expand=True)
            mask = mask.rotate(180, expand=True)
        elif rnd_n < .75:
            image = image.rotate(270, expand=True)
            mask = mask.rotate(270, expand=True)

        new_im = Image.new(image.mode, dims, (0,))
        new_im.paste(image, (0, 0))

        # finalising target
        # patch
        mask = np.array(mask)
        mask[:10, :] = 255
        mask[-10:, :] = 255
        mask[:, :10] = 255
        mask[:, -10:] = 255
        mask = Image.fromarray(mask)
        mask_b = Image.new(mask.mode, output_shape, (255,))
        mask_b.paste(mask, (0, 0))

        return new_im, mask_b

    def calculate_corners(self, mask):
        corners = cv2.goodFeaturesToTrack(mask, 4, 0.01, 5)
        corners = np.reshape(corners, (4, 2))
        if len(corners) != 4:
            np.append(corners, [0, 0])
            np.append(corners, [0, 0])

        corners_ordered = []
        temp = [c[0] + c[1] for c in corners]
        corners_ordered.append(np.argmin(temp))

        while len(corners_ordered) != 4:
            min_angle, idx = np.inf, 0
            oy = np.array([0, 1]) / np.linalg.norm([0, 1])
            for j in range(len(corners)):
                if np.isin(j, corners_ordered):
                    pass
                else:
                    c = corners[j]
                    first_corner = corners[corners_ordered[0]]
                    c_origin = np.array((c[0] - first_corner[0], c[1] - first_corner[1]))
                    unit_c = c_origin / np.linalg.norm(c_origin)
                    dot_product = np.dot(oy, unit_c)
                    angle = np.arccos(dot_product)
                    if angle < min_angle:
                        min_angle = angle
                        idx = j

            corners_ordered.append(idx)
        corners = [corners[j] for j in corners_ordered]
        return corners
