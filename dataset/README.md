# Dataset Directory

The `Dataset` directory is intended to store the dataset used for training the pothole detection model.

## Contents

### `archive.zip`
- **Purpose**: This file contains a large collection of images and corresponding labels used for training the model to detect potholes.
- **Note**: The `archive.zip` file is not included in this repository due to its large size. However, you can download the dataset from the following link:

[Download Pothole Detection Dataset](https://www.kaggle.com/datasets/rajdalsaniya/pothole-detection-dataset)

## Usage
Once you download the dataset, the ingestion functions in the `data_handling` folder will automatically handle the extraction of the contents. Ensure that the downloaded `archive.zip` file is accessible by the script, which will then load the images and labels into the `Extracted_Data` directory structure required for training and prediction.

