# Predictions Directory

The `predictions` directory contains the logic for making predictions using the trained model.

## Contents

### `predict_pothole.py`
- **Purpose**: This script is used to predict the presence of potholes in new images. It employs the trained CNN model to make predictions and visualize the results.
- **Key Features**:
  - Non-Maximum Suppression: This method eliminates redundant overlapping boxes to refine predictions.
  - Visualization: The script displays the input image with bounding boxes around detected potholes for easy interpretation of results.

## Usage
To make predictions on a new image, run the following command:

```bash
python predict_pothole.py --image_path your_image.jpg
