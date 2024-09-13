from model.seq_dcnn import SequentialDcnn
import os
from util.eye_diseases_dataset import EyeDiseaseDataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import itertools


#
sDcnn = SequentialDcnn()
model = sDcnn.loadNewModel()


data_dir = "dataset/input_data"

try:
    dataEyeD = EyeDiseaseDataset(data_dir)
    # # Get splitted data
    valid_df, test_df = dataEyeD.split_data_sample_input()
    print("test dataframe shape: ", test_df)

    # #Get Generators
    batch_size = 10
    valid_gen, test_gen = dataEyeD.create_gens_input(valid_df, test_df, batch_size)

except NameError:
    print('Invalid Input', NameError)

# print("test_gen: ", test_gen)
test_images, test_labels = next(iter(test_gen))
# print("test_images: ", test_images)
print("test_labels: ", test_labels)

# result_predicted = model.predict(test_images)
# print("result; ", result_predicted)

def plot_actual_vs_predicted(model, test_data, num_samples=3):
    # Get a batch of test data
    test_images, test_labels = next(iter(test_data))
    print("test image: ", test_images)
    print("test_labels: ", test_labels)

    predictions = model.predict(test_images)

    class_labels = list(test_gen.class_indices.keys())
    print("class labels: ", class_labels)
    print("Predicted: ", predictions)

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

