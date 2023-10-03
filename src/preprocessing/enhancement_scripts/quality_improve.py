from PIL import Image, ImageEnhance, ImageFilter
import os
import numpy as np
import cv2 as cv
from skimage.exposure import adjust_gamma, adjust_sigmoid


def needs_brightness_adjustment(image):
    # Convert the image to HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Calculate the brightness of the image
    brightness = np.mean(hsv_image[:, :, 2])

    # If the brightness is less than 100, the image needs to be brightened
    if brightness < 100:
        return True
    return False


def _adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def needs_sharpness_enhancement(image):
    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Calculate the Laplacian of the image
    laplacian = cv.Laplacian(gray_image, -1).var()

    # If the Laplacian is less than 100, the image needs to be sharpened
    if laplacian < 100:
        return True
    return False


def _enhance_sharpness(image, factor):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


def needs_noise_reduction(image):
    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Calculate the Laplacian of the image
    laplacian = cv.Laplacian(gray_image, -1).var()

    # If the Laplacian is greater than 100, the image needs to have noise reduced
    if laplacian > 100:
        return True
    return False


def _reduce_noise(image, factor):
    return image.filter(ImageFilter.MedianFilter(size=factor))


def _adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def img_array_enh(image):
    enhance = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    enhance = adjust_gamma(enhance, gamma=1.4, gain=1)
    # enhance = adjust_log(enhance, gain= 1)
    enhance = adjust_sigmoid(enhance, cutoff=0.3, gain=5)
    # enhance = cv.GaussianBlur(enhance, (3,3), 0)
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # flipped_kernel = np.flip(kernel, axis = -1)
    # enhance = cv.filter2D(enhance, -1, flipped_kernel)

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

            image = image.filter(ImageFilter.UnsharpMask(1.7, 3, 2))

            image = _enhance_sharpness(image, 2.0)

            image = _adjust_brightness(image, 0.92)

            # image = _adjust_contrast(image, 1.2)

            # image = _reduce_noise(image, 3)

            image = img_array_enh(np.array(image))

            # Save the improved image
            cv.imwrite(os.path.join(self.output_directory, image_name), image)
            # image.save(os.path.join(self.output_directory, image_name))

            # Print the name of the saved image
            print(f"Saved improved image: {image_name}")
