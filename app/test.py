from app.model.seq_dcnn import SequentialDcnn
import os
from util.eye_diseases_dataset import EyeDiseaseDataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import tensorflow as tf



sDcnn = SequentialDcnn()
model = sDcnn.loadNewModel()

# **Model Structure**
# Start reading dataset
data_dir = "dataset/eye_diseases_original_dataset"
#
try:
    dataEyeD = EyeDiseaseDataset(data_dir)
    # # Get splitted data
    train_df, valid_df, test_df = dataEyeD.split_data()
    print("test dataframe shape: ", test_df)

    #Get Generators
    batch_size = 10
    train_gen, valid_gen, test_gen = dataEyeD.create_gens(train_df, valid_df, test_df, batch_size)

except:
    print('Invalid Input')

print("test_gen: ", test_gen)
test_images, test_labels = next(iter(test_gen))
print("test_images: ", test_images)
print("test_labels: ", test_labels)

# **Evaluate model**
def model_eval(model, test_gen):
    ts_length = len(test_df)
    test_batch_size = test_batch_size = max(
        sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
    test_steps = ts_length // test_batch_size

    train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
    valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
    test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)

    print("Train Loss: ", train_score[0])
    print("Train Accuracy: ", train_score[1])
    print('-' * 20)
    print("Validation Loss: ", valid_score[0])
    print("Validation Accuracy: ", valid_score[1])
    print('-' * 20)
    print("Test Loss: ", test_score[0])
    print("Test Accuracy: ", test_score[1])

def plot_confusion_matrix(cm, classes, normalize= False, title= 'Confusion Matrix', cmap= plt.cm.Blues):
	'''
	This function plot confusion matrix method from sklearn package.
	'''

	plt.figure(figsize= (10, 10))
	plt.imshow(cm, interpolation= 'nearest', cmap= cmap)
	plt.title(title)
	plt.colorbar()

	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation= 45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis= 1)[:, np.newaxis]
		print('Normalized Confusion Matrix')

	else:
		print('Confusion Matrix, Without Normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')

	plt.tight_layout()
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')

def predictTestGens():
    # **Get Predictions**
    preds = model.predict_generator(test_gen)
    y_pred = np.argmax(preds, axis=1)
    print(y_pred)

    #### **Confusion Matrics and Classification Report**
    g_dict = test_gen.class_indices
    classes = list(g_dict.keys())
    # Confusion matrix
    cm = confusion_matrix(test_gen.classes, y_pred)
    plot_confusion_matrix(cm=cm, classes=classes, title='Confusion Matrix')

    # Classification report
    print(classification_report(test_gen.classes, y_pred, target_names=classes))


predictTestGens()

def plot_actual_vs_predicted(model, test_data, num_samples=3):
    # Get a batch of test data
    test_images, test_labels = next(iter(test_data))

    predictions = model.predict(test_images)

    class_labels = list(train_gen.class_indices.keys())
    print("class labels: ", class_labels)

    sample_indices = np.random.choice(range(len(test_images)), num_samples, replace=True)
    # Plot the images with actual and predicted labels
    for i in sample_indices:
        actual_label = class_labels[np.argmax(test_labels[i])]
        predicted_label = class_labels[np.argmax(predictions[i])]
        plt.figure(figsize=(8, 4))
        # Actual Image
        plt.subplot(1, 2, 1)
        plt.imshow(test_images[i].astype(np.uint8))
        plt.title(f'Actual: {actual_label}')
        plt.axis('off')
        # Predicted Image
        plt.subplot(1, 2, 2)
        plt.imshow(test_images[i].astype(np.uint8))
        plt.title(f'Predicted: {predicted_label}')
        plt.axis('off')
        plt.show()


plot_actual_vs_predicted(model, test_gen)