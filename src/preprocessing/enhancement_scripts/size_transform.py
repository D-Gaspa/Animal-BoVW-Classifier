import os
from PIL import Image
from PIL.Image import Resampling


def _resize_image(image_path, output_path, new_width):
    try:
        image = Image.open(image_path)
        w_percent = (new_width / float(image.size[0]))
        new_height = int((float(image.size[1]) * float(w_percent)))
        resized_image = image.resize((new_width, new_height), Resampling.LANCZOS)

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
