# Saved Models Directory

The `saved_models` directory contains the trained models for pothole detection.

## Contents

### `CNN_POTHOLE_MODEL.h5`
- **Purpose**: This file contains the trained weights and architecture of the CNN model saved in HDF5 format.
- **Key Features**:
  - Easy loading: The model can be loaded directly into a TensorFlow/Keras environment for inference.
  - Preserves state: All training progress and configurations are saved for future use.

## Usage
To load the model for predictions or further training, use the following code snippet in your Python script:

```python
from tensorflow.keras.models import load_model

model = load_model('saved_models/CNN_POTHOLE_MODEL.h5')
