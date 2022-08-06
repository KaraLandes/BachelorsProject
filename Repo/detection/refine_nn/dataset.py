import PIL
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from ..corners_nn.dataset import CornerRealBillSet, CornerBillOnBackGroundSet


class RefineBillOnBackGroundSet(CornerBillOnBackGroundSet):

    def form_target(self, image: Image, seed: int) -> tuple:
        im, trg = super().form_target(image, seed)

        image, mask = self.preprocess_image(image, output_shape=(1000, 1000), seed=seed)
        mask = np.array(mask)
        corners = self.calculate_corners(mask)

        trg = list(trg)
        trg.append(np.array(image))
        trg.append(corners)
        return im, trg


class RefineRealBillSet(CornerRealBillSet):
    def form_target(self, image: Image, mask: Image, seed: int, im_name: str):
        im, trg = super().form_target(image.copy(), mask.copy(), seed, im_name)

        image, mask = self.preprocess_image(image, mask, (1000, 1000), im_name, seed=seed)
        mask = np.array(mask)
        corners = self.calculate_corners(mask)

        trg = list(trg)
        trg.append(np.array(image))
        trg.append(corners)
        return im, trg