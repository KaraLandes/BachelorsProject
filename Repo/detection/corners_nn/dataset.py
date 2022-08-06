import PIL
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from PIL import Image, ImageFilter
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from ..data import AbstractBillOnBackGroundSet, AbstractRealBillSet
import cv2


class CornerBillOnBackGroundSet(AbstractBillOnBackGroundSet):

    def form_target(self, image: Image, seed: int) -> tuple:
        output_shape = self.output_shape
        image, mask = self.preprocess_image(image, seed, output_shape)

        mask = np.array(mask)
        mask[mask < 250] = 0
        mask[mask >= 250] = 255
        corners = self.calculate_corners(mask)

        corners_relative = [(c[0] / mask.shape[1], c[1] / mask.shape[0]) for c in corners]

        mask[mask < 255] = 0
        mask[mask == 255] = 1
        # mask_in = np.zeros(mask.shape)
        # mask_in[mask == 0] = 1

        image = np.array(image)
        mask_dims = (int(output_shape[0]/8), int(output_shape[0]/8))
        corners_mask = np.zeros(mask_dims)
        for c in corners_relative:
            corners_32 = (int(c[0] * corners_mask.shape[0]),
                          int(c[1] * corners_mask.shape[0]))
            corners_mask[corners_32[1], corners_32[0]] = 1


        mask = cv2.resize(mask, dsize=mask_dims, interpolation=cv2.INTER_AREA)

        return image, (corners, mask)


class CornerRealBillSet(AbstractRealBillSet):
    def form_target(self, image: Image, mask: Image, seed: int, im_name: str):
        output_shape = self.output_shape
        image, mask = self.preprocess_image(image, mask, output_shape=(400, 400),
                                            im_name=im_name, seed=seed)  # I keep huge shape for better corners
        # edges enhancer
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        # reshape to final
        image.thumbnail(size=output_shape, resample=Image.ANTIALIAS)

        image = np.array(image)
        mask = np.array(mask)
        corners = self.calculate_corners(mask)
        corners_relative = [(c[0] / mask.shape[1], c[1] / mask.shape[0]) for c in corners]

        corners = [[c[0] * output_shape[0], c[1] * output_shape[1]] for c in corners_relative]

        mask[mask < 200] = 0
        mask[mask >= 200] = 1

        mask_in = np.zeros(mask.shape)
        mask_in[mask == 0] = 1

        image = np.array(image)

        mask_dims = (int(output_shape[0] / 8), int(output_shape[0] / 8))
        corners_mask = np.zeros(mask_dims)
        for c in corners_relative:
            corners_32 = (int(c[0] * corners_mask.shape[0]),
                          int(c[1] * corners_mask.shape[0]))
            corners_mask[corners_32[1], corners_32[0]] = 1

        mask_in = cv2.resize(mask_in, dsize=mask_dims, interpolation=cv2.INTER_AREA)

        return image, (corners, mask_in)
