import os
import csv
from datetime import datetime
from .enhancement_pipeline import EnhancementPipeline


def enhancer():
    # Execute the Enhancement Pipeline
    '''
    base_data_dir = os.path.join('..', '..', '..', 'data')
    pipeline = EnhancementPipeline(os.path.join(base_data_dir, 'raw_dataset'),
                                   os.path.join(base_data_dir, 'resized_images'),
                                   os.path.join(base_data_dir, 'enhanced_images'))
    '''
    pipeline = EnhancementPipeline(os.path.join('data\\raw_dataset'),
                                   os.path.join('data\\resized_images'),
                                   os.path.join('data\\enhanced_images'))
    
    evaluation_results = pipeline.execute()
    # evaluation_results, best_enhanced_image, worst_enhanced_image, average_enhanced_image = pipeline.execute()

    # Create a timestamped CSV file to store the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join('..', '..', '..', 'results', 'brisque_scores')
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(results_folder, f'evaluation_results_{timestamp}.csv')

    # Write the BRISQUE scores and improvement percentages to a CSV file
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class Name', 'Image Name', 'Raw BRISQUE Score', 'Resized BRISQUE Score'
                         'Enhanced BRISQUE Score', 'Improvement Percentage (Raw-Resized)',
                         'Improvement Percentage (Resized-Enhanced)', 'Improvement Percentage (Raw-Enhanced)'])
        for result in evaluation_results:
            writer.writerow(result)

    # Save the best, worst and average enhanced images to the visualization_examples folder in the results folder
    visualization_examples_folder = os.path.join('..', '..', '..', 'results', 'visualization_examples')
    os.makedirs(visualization_examples_folder, exist_ok=True)
    # best_enhanced_image.save(os.path.join(visualization_examples_folder, f'best_enhanced_image_{timestamp}.jpg'))
    # worst_enhanced_image.save(os.path.join(visualization_examples_folder, f'worst_enhanced_image_{timestamp}.jpg'))
    # average_enhanced_image.save(os.path.join(visualization_examples_folder, f'average_enhanced_image_{timestamp}.jpg'))

    print(f"Results have been saved to {results_file}")


if __name__ == "__main__":
    main()
