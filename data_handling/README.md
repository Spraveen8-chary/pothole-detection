
### `data_handling/README.md`

# Data Handling Directory

The `data_handling` directory is dedicated to the ingestion and preprocessing of data required for training the CNN model.

## Contents

### `ingest_label_data.py`
- **Purpose**: This script loads and processes the label data associated with the training images. It ensures that the labels are in the correct format for the model.
- **Key Features**:
  - Data validation: The script checks for any discrepancies in the label data to ensure quality.
  - Formatting: Labels are transformed into the structure required for model input.

### `ingest_image_data.py`
- **Purpose**: This script handles the loading of image data from the specified directory and applies necessary preprocessing steps.
- **Key Features**:
  - Image resizing: All images are resized to a standard size for uniformity during training.
  - Data augmentation: Optional transformations can be applied to enhance the dataset.

## Usage
To load your data, run the respective scripts as follows:

```bash
python ingest_label_data.py
python ingest_image_data.py
