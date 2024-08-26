from app.model.seq_dcnn import SequentialDcnn
import os
from util.eye_diseases_dataset import EyeDiseaseDataset
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt



sDcnn = SequentialDcnn()
model = sDcnn.loadNewModel()

model.summary()

try:
    dataDir = '../dataset/eye_diseases_original_dataset/dataset'
    dataSplit = EyeDiseaseDataset(dataDir)
    # Get splitted data
    train_data, valid_data, test_data = dataSplit.split_()
    # print(train_data)
except:
    print('Invalid Input')

# Get Generators
batch_size = 40
train_augmented, valid_augmented, test_augmented = sDcnn.create_gens(train_data, valid_data, test_data, batch_size)
#
print("test_augmented", len(test_augmented), test_augmented)
test_images, test_labels = next(iter(test_augmented))
predictions = model.predict(test_images)
# print("test_images", len(test_images), test_images)
# print("test_labels", len(test_labels), test_labels)
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# print(ROOT_DIR)
# print(os.getcwd())
# print(os.path.join("training_2", ""))

def plot_actual_vs_predicted(model, test_data, num_samples=3):
    # Get a batch of test data
    test_images, test_labels = next(iter(test_data))

    predictions = model.predict(test_images)

    class_labels = list(train_augmented.class_indices.keys())

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


# plot_actual_vs_predicted(model, test_augmented)

#Load one image
predictImage = "input/img.png"
df_predict_image = sDcnn.loadPredictImage(predictImage)
print("df_predict_image", df_predict_image)