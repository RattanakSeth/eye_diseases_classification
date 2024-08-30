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

predictImage = "input/img.png"
image = sDcnn.loadPredictImage(predictImage)
# tensorImage = tf.convert_to_tensor
print("image: ", image.shape)
labels =  ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Fseries = pd.Series([image], name='filepaths')
# Lseries = pd.Series(["cataract"], name='labels')
# data_frame = pd.concat([Fseries, Lseries], axis=1)
# print("data_frame: ", data_frame)
img_size = (224, 224)
channels = 3  # either BGR or Grayscale
color = 'rgb'


def scalar(img):
    return img


# tr_gen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
ts_gen = ImageDataGenerator(preprocessing_function=scalar)
# test_gen = ts_gen.flow_from_dataframe()
#.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
#color_mode=color, shuffle=False, batch_size=1)

model.predict(image)

# valid and test dataframe
# strat = data_frame['labels']
# valid_df_demo, test_df_demo = train_test_split(data_frame, train_size=0.5, shuffle=True, random_state=123, stratify=strat)
# valid_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# test_generator = valid_test_datagen.flow_from_dataframe(
#             data_frame,
#             x_col='filepaths',
#             y_col='labels',
#             target_size=img_size,
#             color_mode=color,
#             batch_size=1,
#             shuffle=False,
#             class_mode='categorical'
#         )
#
# print("Shape of test images:", test_generator.image_shape)

# def define_df(self, files, classes):
#     Fseries = pd.Series(files, name='filepaths')
#     Lseries = pd.Series(classes, name='labels')
#     return pd.concat([Fseries, Lseries], axis=1)
