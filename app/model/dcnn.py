
import os
# from keras.api.applications.efficientnet import EfficientNetB3
from keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.layers import Flatten, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras import Model
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

class DCNN_Model():

    def __init__(self):
        # Save Checkpoint during training
        self.checkpoint_path = os.path.dirname('training_1/cp.weights.h5')#"../../training_1/cp.weights.h5"
        # self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        # self.eye_disease_dataset = EyeDiseaseDataset()
        self.img_size = (224, 224)
        self.channels = 3
        # img_shape = (img_size[0], img_size[1], channels)

    # def create_model(self):
    #     # train_augmented, valid_augmented, test_augmented = self.augment_data(train_data, valid_data, test_data)
    #
    #     classes = len(list(train_augmented.class_indices.keys()))
    #
    #     base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    #
    #     for layer in base_model.layers:
    #         layer.trainable = False
    #     x = base_model.output
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    #
    #     predictions = Dense(classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)
    #
    #     model = Model(inputs=base_model.input, outputs=predictions)
    #
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    #
    #     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #
    #     return model

    def loadNewModel(self):
        return tf.keras.models.load_model(self.checkpoint_path)

    def loadPredictImage(self, picturePath):
        img_width, img_height = self.img_size
        test_image = image.load_img(picturePath, target_size=(img_width, img_height))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image.reshape(self.img_size[0], self.img_size[1])
        # result = model.predict(test_image)
        return test_image

