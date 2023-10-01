import os
from PIL import Image
from PIL.Image import Resampling


def _resize_image(image_path, output_path, min_size):
    try:
        # change the image size depending on the minimum size keeping the aspect ratio
        image = Image.open(image_path)
        width, height = image.size

        # check if the image is already bigger than the minimum size
        if width > min_size and height > min_size:
            return

        if width < height:
            new_width = min_size
            new_height = int(height * (min_size / width))
        else:
            new_height = min_size
            new_width = int(width * (min_size / height))

        # resize the image
        resized_image = image.resize((new_width, new_height), resample=Resampling.LANCZOS)

        # save the resized image
        save_path = os.path.join(output_path, os.path.basename(image_path))
        print(f"Saving to: {save_path}")
        resized_image.save(save_path)
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")


def _create_output_directory(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        raise


class ImageEnhancer:
    def __init__(self, source_directory, output_directory):
        self.source_directory = source_directory
        self.output_directory = output_directory

    def resize_images(self, new_width):
        try:
            _create_output_directory(self.output_directory)

            for image_name in os.listdir(self.source_directory):
                image_path = os.path.join(self.source_directory, image_name)
                _resize_image(image_path, self.output_directory, new_width)
        except Exception as e:
            print(f"Error resizing images in {self.source_directory}: {e}")
