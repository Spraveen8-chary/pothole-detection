# Pothole Detection Project
## Overview
The Pothole Detection Project uses a Convolutional Neural Network (CNN) to identify and localize potholes in images. The aim of this project is to support automated analysis of road conditions, which could aid in infrastructure maintenance and public safety. The system is built to process raw images, detect potholes, and visualize the results by marking potholes with bounding boxes.

## Project Structure
The repository is organized as follows:
<pre>
Pothole Detection Project
├── build_model
│   └── model_building.py      # Code to build and compile the CNN model
├── data_handling
│   ├── ingest_label_data.py    # Ingests and processes label data
│   └── ingest_image_data.py    # Ingests and processes image data
├── predictions
│   └── predict_pothole.py      # Generates predictions and visualizes results
├── saved_models
│   └── CNN_POTHOLE_MODEL.h5    # Saved model weights after training
├── Dataset
│   └── archive.zip             # Link to download the dataset (large files not included)
├── Extracted_Data              # Contains train and validation images and labels after loading
└── README.md                   # Main project README file
</pre>

## Dataset

The dataset used for training and testing is available on Kaggle: [Pothole Detection Dataset](https://www.kaggle.com/datasets/rajdalsaniya/pothole-detection-dataset). Due to its large size, it’s not included in this repository. The data ingestion scripts in `data_handling` will automatically load and extract the dataset into the appropriate structure.

## Features

- **Image Ingestion & Label Processing**: Automatically loads and preprocesses images and labels, creating a structured dataset for training and validation.
- **Model Building**: Utilizes TensorFlow to build a CNN model optimized for pothole detection.
- **Prediction & Visualization**: After training, the model can predict pothole locations in new images, displaying bounding boxes around detected potholes.
- **Model Storage**: Includes a directory for storing trained models, allowing easy reuse for predictions and future training.

## Installation

To run this project, you need Python and several libraries. Install the dependencies with:

```
pip install -r requirements.txt
```
- Make sure to download the dataset from Kaggle and place it in the Dataset directory. Alternatively, provide the dataset URL if integrating with a larger data pipeline.

## Usage
**Training the Model**
- **Run ```model_building.py```**: This script builds, compiles, and trains the CNN model on the pothole dataset. Model checkpoints will be saved in ```saved_models``` for future use.
```
python build_model/model_building.py
```
- **Inspect Training and Validation Losses**: View loss progression using graphs, which will help to monitor training and avoid overfitting.

## Prediction and Visualization
- Use ```predict_pothole.py``` in the ```predictions``` folder to make predictions on new images.
- Bounding boxes around detected potholes will be displayed, providing visual confirmation of the model’s performance.
```
python predictions/predict_pothole.py --image_path /path/to/image.jpg
```

## Model Loss During Training 
- Train Loss VS Validation Loss

![losses](https://github.com/user-attachments/assets/fcc36aa7-ef97-4dcc-887a-07cdfa2565fa)

## PREDICTIONS

**TRUTH**

![truth1](https://github.com/user-attachments/assets/29198f12-cbc2-44d6-b587-d2e7efdff903) 

**PREDICTION**

![pred1](https://github.com/user-attachments/assets/3bc96a74-89e8-4586-8b4e-13b028f6d358) 

**TRUTH**

![truth2](https://github.com/user-attachments/assets/9ac7b222-89da-4068-883d-863563aad6da)

**PREDICTION**

![pred2](https://github.com/user-attachments/assets/d38aae85-a662-404c-9b7c-8d797c1b44d0)











