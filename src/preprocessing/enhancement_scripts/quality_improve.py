from PIL import Image, ImageEnhance, ImageFilter
import os
import numpy as np
import cv2 as cv
from skimage.exposure import adjust_gamma, adjust_sigmoid


def _adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def _enhance_sharpness(image, factor):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


def img_array_enh(image):
    enhanced_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    enhanced_image = adjust_gamma(enhanced_image, gamma=1.4, gain=1)
    enhanced_image = adjust_sigmoid(enhanced_image, cutoff=0.3, gain=5)

    return enhanced_image


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

            image = image.filter(ImageFilter.UnsharpMask(1.7, 3, 2))

            image = _enhance_sharpness(image, 2.0)

            image = _adjust_brightness(image, 0.92)

            image = img_array_enh(np.array(image))

            # Save the improved image
            cv.imwrite(os.path.join(self.output_directory, image_name), image)

            # Print the name of the saved image
            print(f"Saved improved image: {image_name}")
