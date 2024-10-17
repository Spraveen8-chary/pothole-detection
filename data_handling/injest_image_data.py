import os
from abc import ABC, abstractmethod
import zipfile
import cv2
import numpy as np

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> dict:
        """Abstract method to ingest data from a given file."""
        pass

def load_image(image_path: str, target_size: tuple = (256, 256)) -> np.ndarray:

    """Loads and processes a single image."""

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalize to [0, 1]

class ZipDataIngestor(DataIngestor):


    def ingest(self, file_path: str) -> dict:

        """Extracts zip file and loads images from 'train' and 'valid' folders."""
        
        if not file_path.endswith('.zip'):
            raise ValueError('Provided file is not a ".zip" file.')

        # Extract the zip file
        extract_path = os.path.join(os.getcwd(), "Extracted_Data")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        datasets = {}

        # Process 'train' and 'valid' datasets
        for dataset_type in ['train', 'valid']:
            dataset_path = os.path.join(extract_path, dataset_type)
            image_folder = os.path.join(dataset_path, 'images')
            label_folder = os.path.join(dataset_path, 'labels')

            if not os.path.exists(image_folder) or not os.path.exists(label_folder):
                raise FileNotFoundError(f'"images" or "labels" folders not found in {dataset_path}.')

            image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
            label_files = os.listdir(label_folder)

            if len(image_files) == 0 or len(label_files) == 0:
                raise FileNotFoundError(f'No images or labels found in {dataset_type} folder.')

            if len(image_files) != len(label_files):
                raise ValueError(f'Mismatch in number of images and labels in {dataset_type} folder.')

            # Load images
            images = []
            for image_file in image_files:
                try:
                    image_path = os.path.join(image_folder, image_file)
                    image = load_image(image_path)
                    images.append(image)
                except Exception as e:
                    print(f"Error loading {image_file}: {e}")

            datasets[dataset_type] = np.array(images)

        return datasets


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_path: str) -> DataIngestor:
        """Returns the appropriate data ingestor based on file extension."""
        if file_path.endswith('.zip'):
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {os.path.splitext(file_path)[1]}")


if __name__ == '__main__':
    file_path = "..\\pothole detection\\dataset\\archive.zip"
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_path)
    datasets = data_ingestor.ingest(file_path)

    # Access train and valid datasets
    train_images = datasets['train']
    valid_images = datasets['valid']
    print(len(train_images))
