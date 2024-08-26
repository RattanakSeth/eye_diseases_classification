import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
class SequentialDcnn():
    def __init__(self):
        self.checkpoint_path = "../training_2/efficientnetb3-Eye Disease-92.65.h5"
        self.img_size = (224, 224)
        self.channels = 3  # either BGR or Grayscale
        self.color = 'rgb'

    def loadNewModel(self):
        return tf.keras.models.load_model(self.checkpoint_path)

    #### Function to generate images from dataframe
    def create_gens(self, train_df, valid_df, test_df, batch_size):
        '''
        This function takes train, validation, and test dataframe and fit them into image data generator, because model takes data from image data generator.
        Image data generator converts images into tensors. '''

        # define model parameters
        img_size = (224, 224)
        channels = 3  # either BGR or Grayscale
        color = 'rgb'
        img_shape = (img_size[0], img_size[1], channels)

        # Recommended : use custom function for test data batch size, else we can use normal batch size.
        ts_length = len(test_df)
        test_batch_size = max(
            sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
        test_steps = ts_length // test_batch_size

        print("test batch size: ", test_batch_size)

        # This function which will be used in image data generator for data augmentation, it just take the image and return it again.
        def scalar(img):
            return img

        tr_gen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
        ts_gen = ImageDataGenerator(preprocessing_function=scalar)

        train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                               class_mode='categorical',
                                               color_mode=color, shuffle=True, batch_size=batch_size)

        valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                               class_mode='categorical',
                                               color_mode=color, shuffle=True, batch_size=batch_size)

        # Note: we will use custom test_batch_size, and make shuffle= false
        test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                              class_mode='categorical',
                                              color_mode=color, shuffle=False, batch_size=test_batch_size)

        return train_gen, valid_gen, test_gen

    def loadPredictImage(self, picturePath):
        img_width, img_height = self.img_size
        test_image = image.load_img(picturePath, target_size=(img_width, img_height))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image.reshape(self.img_size[0], self.img_size[1], 3)
        print("image width: ", self.img_size[0])
        # result = model.predict(test_image)
        return test_image
