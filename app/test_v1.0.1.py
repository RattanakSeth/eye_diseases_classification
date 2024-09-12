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
# sDcnn = SequentialDcnn()
# model = sDcnn.loadNewModel()


data_dir = "dataset/input_data"

try:
    dataEyeD = EyeDiseaseDataset(data_dir)
    # # Get splitted data
    train_df, valid_df, test_df = dataEyeD.split_data_sample_input()
    print("test dataframe shape: ", test_df)

    # #Get Generators
    # batch_size = 10
    # train_gen, valid_gen, test_gen = dataEyeD.create_gens(train_df, valid_df, test_df, batch_size)

except NameError:
    print('Invalid Input', NameError)

# print("test_gen: ", test_gen)
# test_images, test_labels = next(iter(test_gen))
# print("test_images: ", test_images)
# print("test_labels: ", test_labels)
