from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MAX_POTHOLES = 20

@dataclass
class CNN_Pothole_model:
    X_train: np.ndarray   
    X_val : np.ndarray 
    y_train : np.ndarray 
    y_val : np.ndarray 
    verbose : int = 1
    batch_size: int = 32
    epochs : int = 100
    def ImageDataGeneration(self):
        """Data augmentation configuration."""
        return (ImageDataGenerator(
            rotation_range = 20,
            zoom_range = 0.15,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.15,
            horizontal_flip = True,
            fill_mode = "nearest"
        ))
    
    def build_model(self,input_shape=(256, 256, 3), max_potholes=MAX_POTHOLES):
        """
        Builds a CNN-based object detection model.

        Args:
            input_shape (tuple): Shape of the input images.
            max_potholes (int): Maximum number of potholes per image.

        Returns:
            tensorflow.keras.Model: Compiled CNN model.
        """
        model = models.Sequential()
        
        model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        
        model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        
        model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        
        model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Dense(max_potholes * 5, activation='sigmoid'))
        
        return model
    
    def custom_loss(self,y_true, y_pred):
        """
        Custom loss function combining localization and confidence losses.

        Args:
            y_true (tensor): Ground truth labels.
            y_pred (tensor): Predicted labels.

        Returns:
            tensor: Combined loss.
        """
        y_true = tf.reshape(y_true, (-1, MAX_POTHOLES, 5))
        y_pred = tf.reshape(y_pred, (-1, MAX_POTHOLES, 5))
        
        true_boxes = y_true[:, :, :4]
        true_conf = y_true[:, :, 4]
        
        pred_boxes = y_pred[:, :, :4]
        pred_conf = y_pred[:, :, 4]
        
        loc_loss = tf.reduce_mean(tf.square(true_boxes - pred_boxes) * tf.expand_dims(true_conf, axis=-1))
        
        conf_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(true_conf, pred_conf))
        
        total_loss = loc_loss + conf_loss
        return total_loss

    def train(self):
        """
        Train the model with augmented data.
        """
        # Build model
        model = self.build_model(input_shape=self.X_train.shape[1:])
        
        # Data augmentation
        data_gen = self.ImageDataGeneration()

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=self.custom_loss)
        
        # Train the model using the augmented data
        model.fit(
            data_gen.flow(self.X_train, self.y_train, batch_size=self.batch_size),
            validation_data=(self.X_val, self.y_val),
            epochs=self.epochs,
            verbose=self.verbose,
            steps_per_epoch=len(self.X_train) // self.batch_size,
        )
        
        return model
    
