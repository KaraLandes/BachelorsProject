import os


import numpy as np
import warnings

warnings.filterwarnings("ignore")
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from ..data import AbstractBillOnBackGroundSet, AbstractRealBillSet

class BaseBillOnBackGroundSet(AbstractBillOnBackGroundSet):
    """
    Generated Bills dataset.
    """

    def form_target(self, image: Image, seed: int) -> tuple:
        """
        This method runs unified preprocessing procedure
        and then forms target w.r.t. the network structure
        :param image: Image of the bill
        :param seed: seeding random generator
        :param output_shape: shape of final image and mask for inputs unification
        :return: tuple (array of image, specific target)
        """
        output_shape = self.output_shape
        image, mask = self.preprocess_image(image, seed, output_shape)

        mask = np.array(mask)
        mask[mask < 200] = 0
        mask[mask >= 200] = 256
        rows = np.argwhere(np.isclose(mask, 256))[:, 0]
        columns = np.argwhere(np.isclose(mask, 256))[:, 1]
        target = []  # target array collects all transformations done to image in order to revert t afterwards
        target.append(min(rows)/image.height)
        target.append(max(rows)/image.height)
        target.append(min(columns)/image.width)
        target.append(max(columns)/image.width)

        return np.array(image), target




class BaseRealBillSet(AbstractRealBillSet):

    def form_target(self, image: Image, mask: Image, seed: int):
        output_shape = self.output_shape
        image, mask = self.preprocess_image(image, mask, output_shape)

        mask = np.array(mask)
        rows = np.argwhere(mask == 0)[:, 0]
        columns = np.argwhere(mask == 0)[:, 1]

        target = []  # target array collects all transformations done to image in order to revert t afterwards
        target.append(min(rows)/image.height)
        target.append(max(rows)/image.height)
        target.append(min(columns)/image.width)
        target.append(max(columns)/image.width)

        return np.array(image), target
