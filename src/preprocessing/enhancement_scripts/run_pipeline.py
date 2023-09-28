import os
import csv
from datetime import datetime
from enhancement_pipeline import EnhancementPipeline


def main():
    # Execute the Enhancement Pipeline
    pipeline = EnhancementPipeline('../../../data/raw_dataset',
                                   '../../../data/resized_images',
                                   '../../../data/enhanced_images')
    evaluation_results = pipeline.execute()

    # Create a timestamped CSV file to store the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = '../../../results/brisque_scores'
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(results_folder, f'evaluation_results_{timestamp}.csv')

    # Write the BRISQUE scores and improvement percentages to a CSV file
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class Name', 'Image Name', 'Raw BRISQUE Score',
                         'Enhanced BRISQUE Score', 'Improvement Percentage'])

        for result in evaluation_results:
            writer.writerow(result)

    print(f"Results have been saved to {results_file}")


if __name__ == "__main__":
    main()
