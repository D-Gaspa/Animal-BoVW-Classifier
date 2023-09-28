from PIL import Image, ImageEnhance, ImageFilter
import os


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
            image = _adjust_contrast(image, 1.2)
            image = _reduce_noise(image, 3)
            image = _enhance_sharpness(image, 2.0)
            image = _adjust_brightness(image, 1.1)

            # Save the improved image
            image.save(os.path.join(self.output_directory, image_name))

            # Print the name of the saved image
            print(f"Saved improved image: {image_name}")
