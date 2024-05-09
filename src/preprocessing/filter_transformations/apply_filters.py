import os

import cv2

from filters import Filters


class ApplyFilters:
    def __init__(self, input_folder, output_folder, filters):
        self.input_folder = input_folder
        self.output_folder = os.path.join(output_folder, "-".join(filters))
        self.filters = filters

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def apply(self):
        # This dictionary maps the filter name to the corresponding function
        filter_methods = {
            "histogram_equalization": Filters.histogram_equalization,
            "clahe": Filters.clahe,
            "edge_enhancement": Filters.edge_enhancement,
            "noise_reduction": Filters.noise_reduction,
            "sharpen": Filters.sharpen,
            "edge_detection": Filters.edge_detection,
            "color_filtering": Filters.color_filtering,
            "adaptive_thresholding": Filters.adaptive_thresholding,
            "gabor": Filters.gabor,
            "background_removal": Filters.background_removal,
            "canny_edge_detection": Filters.canny_edge_detection,
            "unsharp_masking": Filters.unsharp_masking
        }

        print("Applying filters")

        # Iterate over all the classes
        for class_folder in os.listdir(self.input_folder):
            class_path = os.path.join(self.input_folder, class_folder)

            # Create the class folder in the output folder
            filtered_class_folder = os.path.join(self.output_folder, class_folder)

            print(f"Applying filters to class {class_folder}")

            if not os.path.exists(filtered_class_folder):
                os.makedirs(filtered_class_folder)

            # Iterate over all the images in the class folder
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image = cv2.imread(image_path)

                print(f"Applying filters to image {image_name} in class {class_folder}")

                # Apply all the filters in the order specified
                for filter_name in self.filters:
                    filter_function = filter_methods[filter_name]
                    image = filter_function(image)

                # Save the filtered image
                cv2.imwrite(os.path.join(filtered_class_folder, image_name), image)
                print(f"Applied filters to image {image_name} in class {class_folder}")


if __name__ == "__main__":
    base_data_dir = os.path.join('..', '..', '..', 'data')
    enhanced_images_folder = os.path.join(base_data_dir, 'enhanced_images')
    resized_images_folder = os.path.join(base_data_dir, 'resized_images')
    filtered_images_folder = os.path.join(base_data_dir, 'filtered_images')
    background_removed_images_folder = os.path.join(filtered_images_folder, 'background_removal')

    # Specify the filters and their order in this list
    filters_to_apply = ["unsharp_masking"]

    apply_filters = ApplyFilters(background_removed_images_folder, filtered_images_folder, filters_to_apply)
    apply_filters.apply()
