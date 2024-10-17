# Build Model Directory

The `build_model` directory contains all the scripts related to building the Convolutional Neural Network (CNN) model for pothole detection.

## Contents

### `model_building.py`
- **Purpose**: This script is responsible for defining the architecture of the CNN. It includes the layers, activation functions, and the compilation process.
- **Key Features**:
  - Customizable architecture: Users can adjust the number of layers and their types (e.g., convolutional, pooling).
  - Model compilation options: You can specify different optimizers and loss functions based on your requirements.
  
## Usage
To build the model, simply run the `model_building.py` script. Ensure that you have all the necessary dependencies installed as specified in the `requirements.txt` file.

```bash
python model_building.py
