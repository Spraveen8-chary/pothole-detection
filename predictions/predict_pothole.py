from typing import List
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_handling.injest_image_data import load_image


MAX_POTHOLES = 20
IMAGE_SIZE = (256, 256)

class Prediction:
    def non_max_suppression(self, boxes:np.ndarray, confidences:np.ndarray, iou_threshold:float=0.5)->List[int]:
        """
        Applies Non-Maximum Suppression to eliminate redundant overlapping boxes.

        Args:
            boxes (np.ndarray): Array of bounding boxes [x1, y1, x2, y2].
            confidences (np.ndarray): Array of confidence scores.
            iou_threshold (float): IoU threshold for suppression.

        Returns:
            List[int]: Indices of boxes to keep.
        """
        if len(boxes) == 0:
            return []
        boxes_tensor = tf.constant(boxes, dtype=tf.float32)
        confidences_tensor = tf.constant(confidences, dtype=tf.float32)
        selected_indices = tf.image.non_max_suppression(
            boxes_tensor,
            confidences_tensor,
            max_output_size=MAX_POTHOLES,
            iou_threshold=iou_threshold
        )
        return selected_indices.numpy()
    
    def process_predictions(self, predictions:np.ndarray, threshold:float=0.5)->List[List[List[int]]]:
        """
        Processes raw model predictions to extract bounding boxes.

        Args:
            predictions (np.ndarray): Raw predictions from the model.
            threshold (float): Confidence threshold to filter boxes.

        Returns:
            List[List[List[int]]]: List of bounding boxes for each image.
        """
        processed_boxes = []
        for pred in predictions:
            boxes = []
            confidences = []
            for i in range(MAX_POTHOLES):
                start = i * 5
                x_center, y_center, width, height, conf = pred[start:start+5]
                if conf > threshold:
                    x_center_abs = x_center * IMAGE_SIZE[1]
                    y_center_abs = y_center * IMAGE_SIZE[0]
                    width_abs = width * IMAGE_SIZE[1]
                    height_abs = height * IMAGE_SIZE[0]
                    
                    x1 = int(x_center_abs - width_abs / 2)
                    y1 = int(y_center_abs - height_abs / 2)
                    x2 = int(x_center_abs + width_abs / 2)
                    y2 = int(y_center_abs + height_abs / 2)
                    
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(conf)
            
            boxes = np.array(boxes)
            confidences = np.array(confidences)
            
            if len(boxes) == 0:
                processed_boxes.append([])
                continue
            
            selected_indices = self.non_max_suppression(boxes, confidences, iou_threshold=0.3)
            selected_boxes = boxes[selected_indices]
            
            processed_boxes.append(selected_boxes.tolist())
        
        return processed_boxes
    
    def visualize_predictions(self, image:np.ndarray, boxes:List[List[int]], title:str="Predictions"):
        """
        Displays an image with bounding boxes.

        Args:
            image (np.ndarray): Image array.
            boxes (List[List[int]]): List of bounding boxes [x1, y1, x2, y2].
            title (str): Title of the plot.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        ax = plt.gca()
        
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        plt.title(title)
        plt.axis('off')
        plt.show()

    def predict(self, model:tf.keras.Model, image_path:str, threshold:float=0.5):
        """
        Predicts potholes in an image and visualizes the results.

        Args:
            model (tensorflow.keras.Model): Trained model.
            image_path (str): Path to the image file.
            threshold (float): Confidence threshold for predictions.
        """
        image = load_image(image_path)
        image_input = np.expand_dims(image, axis=0)
        
        prediction = model.predict(image_input)
        
        predicted_boxes = self.process_predictions(prediction, threshold=threshold)[0]
        
        image_disp = (image * 255).astype(np.uint8)
        
        self.visualize_predictions(image_disp, predicted_boxes, title="Predicted Potholes")

