import os
import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.exposure import adjust_gamma, adjust_sigmoid


def _adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def _enhance_sharpness(image, factor):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


def _reduce_noise(image, factor):
    return image.filter(ImageFilter.MedianFilter(size=factor))


def _adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def imgarr_enh(image):
    enhance = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    enhance = adjust_gamma(enhance, gamma=1.4, gain=1)
    # enhance = adjust_log(enhance, gain= 1)
    enhance = adjust_sigmoid(enhance, cutoff=0.3, gain=5)
    # enhance = unsharp_mask(enhance, radius=20, amount = 1.0)
    # enhance = gray2rgb(enhance)
    # kernel = np.array(([1, -2, 1], [-2, 4, -2], [1, -2, 1]), np.float32) / 9
    # flipped_kernel = np.flip(kernel, axis = -1)
    # unsharp_masking = cv.filter2D(image, -1, flipped_kernel)
    return enhance


class QualityImprover:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    def improve_quality(self):
        for image_name in os.listdir(self.input_directory):
            image_path = os.path.join(self.input_directory, image_name)
            image = Image.open(image_path)

            # Apply improvements here
            # image = _adjust_contrast(image, 1.5)
            # image = _reduce_noise(image, 3)
            image = _enhance_sharpness(image, 2.0)
            image = _adjust_brightness(image, 0.92)
            image = imgarr_enh(np.array(image))

            # Save the improved image
            cv.imwrite(os.path.join(self.output_directory, image_name), image)
            # image.save(os.path.join(self.output_directory, image_name))

            # Print the name of the saved image
            print(f"Saved improved image: {image_name}")
