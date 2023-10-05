import os
import random
import numpy as np
from PIL import Image
from brisque import BRISQUE
from skimage import color
from .size_transform import ImageEnhancer
from .quality_improve import QualityImprover


class BrisqueEvaluator:
    def __init__(self, raw_images_path, resized_images_path, enhanced_images_path):
        self.raw_images_path = raw_images_path
        self.resized_images_path = resized_images_path
        self.enhanced_images_path = enhanced_images_path

    def evaluate(self):
        # return the evaluation results, best, worst and average enhanced images
        obj = BRISQUE(url=False)
        evaluation_results = []
        negative_brisque_scores = []
        enhanced_images_and_scores = []

        best_enhanced_image = None
        best_enhanced_score = float('-inf')
        worst_enhanced_image = None
        worst_enhanced_score = float('inf')

        total_raw_score = 0
        total_resized_score = 0
        total_enhanced_score = 0
        total_images = 0

        # Iterate through each class directory
        for class_name in os.listdir(self.enhanced_images_path):
            raw_class_directory = os.path.join(self.raw_images_path, class_name)
            resized_class_directory = os.path.join(self.resized_images_path, class_name)
            enhanced_class_directory = os.path.join(self.enhanced_images_path, class_name)

            all_images = os.listdir(enhanced_class_directory)
            selected_images = random.sample(all_images, 10)  # Selecting 10 images from each class
            # selected_images = all_images

            # Iterate through each image in the class directory
            for image_name in selected_images:
                try:
                    raw_image_path = os.path.join(raw_class_directory, image_name)
                    resized_image_path = os.path.join(resized_class_directory, image_name)
                    enhanced_image_path = os.path.join(enhanced_class_directory, image_name)

                    raw_image = np.array(Image.open(raw_image_path))
                    resized_image = np.array(Image.open(resized_image_path))
                    enhanced_image = np.array(Image.open(enhanced_image_path))

                    # Check if the image is grayscale; if so, convert it to RGB
                    if len(raw_image.shape) == 2:
                        raw_image = color.gray2rgb(raw_image)
                    if len(resized_image.shape) == 2:
                        resized_image = color.gray2rgb(resized_image)
                    if len(enhanced_image.shape) == 2:
                        enhanced_image = color.gray2rgb(enhanced_image)

                    # Calculate the BRISQUE scores for the raw, resized and enhanced images
                    raw_score = obj.score(raw_image)
                    resized_score = obj.score(resized_image)
                    enhanced_score = obj.score(enhanced_image)

                    # Don't count the image if the BRISQUE score is negative
                    if raw_score < 0 or resized_score < 0 or enhanced_score < 0:
                        negative_brisque_scores.append((class_name, image_name))
                        print(f"Skipping image {image_name} in class {class_name} because of negative BRISQUE score.")
                        continue

                    # Save the best and worst enhanced images
                    if enhanced_score > best_enhanced_score:
                        best_enhanced_score = enhanced_score
                        best_enhanced_image = Image.open(enhanced_image_path)
                    if enhanced_score < worst_enhanced_score:
                        worst_enhanced_score = enhanced_score
                        worst_enhanced_image = Image.open(enhanced_image_path)

                    total_images += 1
                    total_raw_score += raw_score
                    total_resized_score += resized_score
                    total_enhanced_score += enhanced_score

                    enhanced_images_and_scores.append((enhanced_image, enhanced_score))

                    # Calculate the improvement percentage; if the enhanced score is lower than the raw score,
                    # there was an improvement in quality
                    improvement_percentage_resized = ((raw_score - resized_score) / abs(resized_score) * 100) \
                        if resized_score != 0 else 0
                    improvement_percentage_enhanced = ((resized_score - enhanced_score) / abs(enhanced_score) * 100) \
                        if enhanced_score != 0 else 0
                    improvement_percentage_overall = ((raw_score - enhanced_score) / abs(enhanced_score) * 100) \
                        if enhanced_score != 0 else 0

                    # Append the scores to the evaluation results
                    evaluation_results.append((
                        class_name, image_name,
                        raw_score, resized_score, enhanced_score,
                        improvement_percentage_resized, improvement_percentage_enhanced
                    ))

                    # Print the scores
                    print(f"Class: {class_name}, Image: {image_name}, "
                          f"Raw Score: {raw_score}, Resized Score: {resized_score}, Enhanced Score: {enhanced_score}, "
                          f"Improvement Percentage (Raw-Resized): {improvement_percentage_resized}, "
                          f"Improvement Percentage (Resized-Enhanced): {improvement_percentage_enhanced}, "
                          f"Improvement Percentage (Raw-Enhanced): {improvement_percentage_overall}")

                except Exception as e:
                    print(f"Error processing image {image_name} in class {class_name}: {e}")

        # Calculate the average scores for the entire dataset
        average_raw_score = total_raw_score / total_images if total_images != 0 else 0
        average_resized_score = total_resized_score / total_images if total_images != 0 else 0
        average_enhanced_score = total_enhanced_score / total_images if total_images != 0 else 0
        average_improvement_percentage_resized = ((average_raw_score - average_resized_score) /
                                                  average_resized_score * 100) if average_resized_score != 0 else 0
        average_improvement_percentage_enhanced = ((average_resized_score - average_enhanced_score) /
                                                   average_enhanced_score * 100) if average_enhanced_score != 0 else 0
        average_improvement_percentage_overall = ((average_raw_score - average_enhanced_score) /
                                                  average_enhanced_score * 100) if average_enhanced_score != 0 else 0

        # Calculate the average enhanced image
        average_enhanced_image = min(enhanced_images_and_scores,
                                     key=lambda x: abs(x[1] - average_enhanced_score))[0]
        # Convert the image to a PIL Image
        average_enhanced_image = Image.fromarray((average_enhanced_image * 255).astype(np.uint8))

        # Append the average scores to the evaluation results
        evaluation_results.append((
            'Average', 'NA',
            average_raw_score, average_resized_score, average_enhanced_score,
            average_improvement_percentage_resized, average_improvement_percentage_enhanced,
            average_improvement_percentage_overall
        ))

        # Print the average scores
        print(f"Average Raw Score: {average_raw_score}, Average Resized Score: {average_resized_score}, "
              f"Average Enhanced Score: {average_enhanced_score}, "
              f"Average Improvement Percentage (Raw-Resized): {average_improvement_percentage_resized}, "
              f"Average Improvement Percentage (Resized-Enhanced): {average_improvement_percentage_enhanced}, "
              f"Average Improvement Percentage (Raw-Enhanced): {average_improvement_percentage_overall}")

        # Print the images with negative BRISQUE scores
        print(f"Images with negative BRISQUE scores: {negative_brisque_scores}")

        return evaluation_results, best_enhanced_image, worst_enhanced_image, average_enhanced_image


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
            enhancer.resize_images(500)

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
        brisque_evaluator = BrisqueEvaluator(self.raw_dataset_path, self.resized_images_path, self.enhanced_images_path)

        return brisque_evaluator.evaluate()
