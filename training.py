from data_handling.ingest_label_data import LabelData
from data_handling.injest_image_data import DataIngestorFactory
from sklearn.model_selection import train_test_split
from build_model.model_building import CNN_Pothole_model
from predictions.predict_pothole import Prediction


dataset_path = '..\\pothole detection\\dataset\\archive.zip'
img_loader = DataIngestorFactory().get_data_ingestor(dataset_path)
images = img_loader.ingest(dataset_path)['train']

labels = LabelData().load_all_labels_for_images()


print(len(images.shape))
print(len(labels.shape))


X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=20)

print(f'''
X_train : {X_train.shape[0]}
X_val   : {X_val.shape[0]}
y_train : {y_train.shape}
y_val   : {y_val.shape}
''')

cnn_model = CNN_Pothole_model(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

# Train the model
model, history = cnn_model.train()

# Print the model summary
print(model.summary())

# Visualizing LOSS vs VAL_LOSS
cnn_model.loss_vs_val_loss(history.history['loss'], history.history['val_loss'])

# sample prediction


img = '..\\pothole detection\\Extracted_Data\\valid\\images\\8_jpg.rf.1c96f62d936a4c9f0d6a49bce97bf010.jpg'

Prediction.predict(model, img, 0.5)

