import os
import random
import numpy as np
from PIL import Image
from brisque import BRISQUE
from skimage import color
from size_transform import ImageEnhancer
from quality_improve import QualityImprover


class BrisqueEvaluator:
    def __init__(self, raw_images_path, enhanced_images_path):
        self.raw_images_path = raw_images_path
        self.enhanced_images_path = enhanced_images_path

    def evaluate(self):
        obj = BRISQUE(url=False)
        evaluation_results = []

        total_raw_score = 0
        total_enhanced_score = 0
        total_images = 0
        negative_raw_score_count = 0
        negative_enhanced_score_count = 0

        # Iterate through each class directory
        for class_name in os.listdir(self.enhanced_images_path):
            enhanced_class_directory = os.path.join(self.enhanced_images_path, class_name)
            raw_class_directory = os.path.join(self.raw_images_path, class_name)

            all_images = os.listdir(enhanced_class_directory)
            # selected_images = random.sample(all_images, 10)  # Selecting 10 images from each class
            selected_images = all_images

            # Iterate through each image in the class directory
            for image_name in selected_images:
                try:
                    enhanced_image_path = os.path.join(enhanced_class_directory, image_name)
                    raw_image_path = os.path.join(raw_class_directory, image_name)

                    enhanced_image = np.array(Image.open(enhanced_image_path))
                    raw_image = np.array(Image.open(raw_image_path))

                    # Check if the image is grayscale; if so, convert it to RGB
                    if len(enhanced_image.shape) == 2:
                        enhanced_image = color.gray2rgb(enhanced_image)
                    if len(raw_image.shape) == 2:
                        raw_image = color.gray2rgb(raw_image)

                    enhanced_score = obj.score(enhanced_image)
                    raw_score = obj.score(raw_image)

                    # If the score is negative, we mark it as "NA"
                    if raw_score < 0 or enhanced_score < 0:
                        negative_raw_score_count += 1
                        negative_enhanced_score_count += 1
                        print(f"Negative score detected. Raw Score: {raw_score}, Enhanced Score: {enhanced_score}")
                        evaluation_results.append((class_name, image_name, 'NA', 'NA', 'NA'))
                        continue

                    total_images += 1
                    total_raw_score += raw_score
                    total_enhanced_score += enhanced_score

                    # Calculate the improvement percentage; if the enhanced score is lower than the raw score,
                    # there was an improvement in quality
                    improvement_percentage = ((raw_score - enhanced_score) / enhanced_score * 100) \
                        if enhanced_score != 0 else 0  # Avoid division by zero

                    evaluation_results.append((class_name, image_name, raw_score, enhanced_score,
                                               improvement_percentage))

                    print(f"Class: {class_name}, Image: {image_name}, Raw Score: {raw_score}, "
                          f"Enhanced Score: {enhanced_score}, Improvement Percentage: {improvement_percentage}")

                except Exception as e:
                    print(f"Error processing image {image_name} in class {class_name}: {e}")

        # Calculate the average scores for the entire dataset
        average_raw_score = total_raw_score / total_images if total_images != 0 else 0
        average_enhanced_score = total_enhanced_score / total_images if total_images != 0 else 0
        average_improvement_percentage = ((average_enhanced_score - average_raw_score) / abs(average_raw_score) * 100) \
            if average_raw_score != 0 else 0

        # Append the average scores to the evaluation results
        evaluation_results.append(("Average", "N/A", average_raw_score, average_enhanced_score,
                                   average_improvement_percentage))

        print(f"Average Raw Score: {average_raw_score}, Average Enhanced Score: {average_enhanced_score}, "
              f"Average Improvement Percentage: {average_improvement_percentage}")

        print(f"Negative Raw Score Count: {negative_raw_score_count}")
        print(f"Negative Enhanced Score Count: {negative_enhanced_score_count}")

        return evaluation_results


class EnhancementPipeline:
    def __init__(self, raw_dataset_path, resized_images_path, enhanced_images_path):
        self.raw_dataset_path = raw_dataset_path
        self.resized_images_path = resized_images_path
        self.enhanced_images_path = enhanced_images_path

    def execute(self):
        if not os.path.exists(self.resized_images_path):
            self._resize_images()

        self._improve_images_quality()
        return self._brisque_evaluation()

    def _resize_images(self):
        print("Resizing images...")

        for class_name in os.listdir(self.raw_dataset_path):
            class_path = os.path.join(self.raw_dataset_path, class_name)
            resized_class_path = os.path.join(self.resized_images_path, class_name)
            os.makedirs(resized_class_path, exist_ok=True)  # Creating class directory in resized_images

            enhancer = ImageEnhancer(class_path, resized_class_path)
            enhancer.resize_images(1000)

        print("Images resized successfully.")

    def _improve_images_quality(self):
        print("Improving image quality...")

        for class_name in os.listdir(self.resized_images_path):
            resized_class_path = os.path.join(self.resized_images_path, class_name)
            enhanced_class_path = os.path.join(self.enhanced_images_path, class_name)
            os.makedirs(enhanced_class_path, exist_ok=True)  # Creating class directory in enhanced_images

            quality_improver = QualityImprover(resized_class_path, enhanced_class_path)
            quality_improver.improve_quality()

        print("Image quality improved successfully.")

    def _brisque_evaluation(self):
        brisque_evaluator = BrisqueEvaluator(self.raw_dataset_path, self.enhanced_images_path)

        return brisque_evaluator.evaluate()
