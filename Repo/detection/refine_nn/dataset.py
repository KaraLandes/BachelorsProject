import PIL
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from PIL import Image, ImageFilter
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from ..corners_nn.dataset import CornerRealBillSet, CornerBillOnBackGroundSet


class RefineBillOnBackGroundSet(CornerBillOnBackGroundSet):

    def form_target(self, image: Image, seed: int) -> tuple:
        im, trg = super().form_target(image, seed)

        image, mask = self.preprocess_image(image, output_shape=(1000, 1000), seed=seed)

        temp = np.zeros((im.shape[1], im.shape[1], 3))
        temp[:, :, 0] = im[0]
        temp[:, :, 1] = im[1]
        temp[:, :, 2] = im[2]
        im = np.array(Image.fromarray(np.array(temp).astype(np.uint8)).convert("L"))

        image = image.convert("RGB").filter(ImageFilter.UnsharpMask(radius=2, percent=50))
        image = np.array(image)
        image = np.array([image[:, :, 0], image[:, :, 1], image[:, :, 2]])

        # thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
        # thresh = 255 - thresh
        mask = np.array(mask)
        corners = self.calculate_corners(mask)

        trg = list(trg)
        trg[1] = mask
        trg.append(image)
        trg.append(corners)
        return im, trg


class RefineRealBillSet(CornerRealBillSet):
    def form_target(self, image: Image, mask: Image, seed: int, im_name: str):
        im, trg = super().form_target(image.copy(), mask.copy(), seed, im_name)

        temp = np.zeros((im.shape[1], im.shape[1], 3))
        temp[:, :, 0] = im[0]
        temp[:, :, 1] = im[1]
        temp[:, :, 2] = im[2]
        im = np.array(Image.fromarray(np.array(temp).astype(np.uint8)).convert("L"))

        image, mask = self.preprocess_image(image, mask, (1000, 1000), im_name, seed=seed)

        image = image.convert("RGB").filter(ImageFilter.UnsharpMask(radius=2, percent=50))
        image = np.array(image)
        image = np.array([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
        # thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
        # thresh = 255 - thresh

        mask = np.array(mask)
        corners = self.calculate_corners(mask)

        trg = list(trg)
        trg[1] = mask
        trg.append(image)
        trg.append(corners)
        return im, trg