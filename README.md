# Animal Image Classification Using BoVW Technique

This repository focuses on the classification of animal images by utilizing the Bag-of-Visual-Words (BoVW) technique. The project incorporates a multi-stage preprocessing pipeline, which includes image enhancement and application of various filters. The BRISQUE algorithm is used to assess the quality of processed images.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)

## Project Overview

- **Objective:** Classify animal images using the BoVW technique.
- **Dataset:** (TODO - Include details about the dataset we're using, or a link to the source).
- **Preprocessing:** Image enhancement for improved visualization, application of various filters to optimize classification accuracy.
- **Assessment:** Employ the [BRISQUE algorithm](https://github.com/rehanguha/brisque) to evaluate 50 random images for enhancement quality.

## Directory Structure
```plaintext
|-- Animal-BoVW-Classifier
    |-- README.md
    |-- LICENSE
    |-- .gitignore
    |
    |-- src
    |   |-- preprocessing
    |   |   |-- enhancement_scripts
    |   |   |   |-- size_transform.py
    |   |   |   |-- quality_improve.py
    |   |   |
    |   |   |-- filter_transformations
    |   |       |-- filter1.py
    |   |       |-- filter2.py
    |   |       |-- ...
    |   |
    |   |-- classification
    |   |   |-- bovw.py
    |   |   |-- feature_extraction.py
    |   |
    |   |-- utilities
    |       |-- utils.py
    |
    |-- data
    |   |-- raw_dataset
    |   |   |-- class1
    |   |   |-- class2
    |   |   |-- ...
    |   |
    |   |-- enhanced_images
    |   |   |-- class1
    |   |   |-- class2
    |   |   |-- ...
    |   |
    |   |-- filtered_images
    |       |-- filter1
    |       |   |-- class1
    |       |   |-- class2
    |       |   |-- ...
    |       |
    |       |-- filter2
    |       |-- ...
    |
    |-- results
    |   |-- brisque_scores
    |   |   |-- brisque_evaluation.txt
    |   |
    |   |-- visualization_examples
    |   |   |-- best_image.jpg
    |   |   |-- average_image.jpg
    |   |   |-- worst_image.jpg
    |   |
    |   |-- classification_results
    |       |-- results.txt
    |
    |-- docs
        |-- project_report.pdf
        |-- ablation_study.md
        |-- ...
```

## Getting Started

### Prerequisites
TODO 
- Software/Tool 1 (e.g., Python 3.x)
- Software/Tool 2 (e.g., OpenCV)
- ...

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/D-Gaspa/Animal-BoVW-Classifier.git
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

(TODO - Adjust these steps later.)

## Usage

TODO - Briefly explain how to run the scripts or use the project, e.g.,

```
python src/preprocessing/enhancement_scripts/size_transform.py --input data/raw_dataset --output data/enhanced_images
```

## Results

For detailed results and visualization examples, refer to the `results` directory.

## Contributors

- Diego Gasparis ([@D-Gaspa](https://github.com/D-Gaspa))
- Mario Sánchez ([@MegaChestercat](https://github.com/MegaChestercat))

## Acknowledgements
- TODO
