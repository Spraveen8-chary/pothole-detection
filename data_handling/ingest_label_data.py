import os
import numpy as np


class LabelData:
    @staticmethod
    def load_labels_for_image(image_filename:str, label_folder:str, max_potholes:int = 20)->np.ndarray:
        
        """Load all labels associated with a single image, up to a max number of potholes."""

        base_name = os.path.splitext(image_filename)[0]
    
        label_files = [f for f in os.listdir(label_folder) if f.startswith(base_name) and f.endswith('.txt')]
        
        boxes = []
        for label_file in label_files:
            label_path = os.path.join(label_folder, label_file)
            with open(label_path, 'r') as file:
                line = file.readline().strip()
                if not line:
                    continue  
                values = list(map(float, line.split()))
                if len(values) != 5:
                    print(f"Warning: Incorrect label format in {label_file}. Expected 5 values, got {len(values)}.")
                    continue  
                class_id, x_center, y_center, width, height = values
                boxes.append([x_center, y_center, width, height, 1.0])  # 1.0 for confidence
        
        num_potholes = len(boxes)
        if num_potholes > max_potholes:
            print(f"Warning: Image {image_filename} has more potholes ({num_potholes}) than MAX_POTHOLES ({max_potholes}). Truncating.")
            boxes = boxes[:max_potholes]
        elif num_potholes < max_potholes:
            for _ in range(max_potholes - num_potholes):
                boxes.append([0, 0, 0, 0, 0.0])  # 0.0 confidence for padding
        
        return np.array(boxes).flatten()
    
    @staticmethod
    def load_all_labels_for_images(data_type='train', max_potholes: int = 20) -> np.ndarray:

        """Load labels for all images in either 'train' or 'valid' dataset."""
        
        dataset_path = os.getcwd()
        folder = os.path.join(dataset_path, 'Extracted_Data', data_type)
        img_folder = os.path.join(folder, 'images')
        label_folder = os.path.join(folder, 'labels')

        image_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png'))]
        labels = []

        for img_file in image_files:
            try:
                label = LabelData.load_labels_for_image(img_file, label_folder, max_potholes)
                labels.append(label)
            except Exception as e:
                print(f"Error loading labels for image {img_file}: {e}")

        return np.array(labels)

if __name__ == '__main__':

    obj = LabelData.load_all_labels_for_image()
    print(len(obj))
