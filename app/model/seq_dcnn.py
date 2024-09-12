import os
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import keras
# from keras._tf_keras.keras.
# from tensorflow.keras.preprocessing import image
# from tensorflow._api.v2.
import numpy as np
# from tensorflow.python.keras.optimizer_v1 import Adam, Adamax
# from tensorflow.python.keras import models

# from tensorflow.python.keras

print(tf.version.VERSION)
class SequentialDcnn():
    def __init__(self):
        self.checkpoint_path = 'training/efficientnetb3-Eye-Disease-92.65.h5'
        self.checkpoint_weight = 'training/efficientnetb3EyeDisease-weights.h5'
        self.img_size = (224, 224)
        self.channels = 3  # either BGR or Grayscale
        self.color = 'rgb'

    def loadNewModel(self):
        # model = tf..load_model(self.checkpoint_path)
        # model = models.load_model(self.checkpoint_path, cru)
        # print("model: ", tf.__version__)
        # tf.keras.models
        # model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        # return model
        #### **Generic Model Creation**
        # Create Model Structure
        img_size = (224, 224)
        channels = 3
        img_shape = (img_size[0], img_size[1], channels)
        # class_count = len(list(train_gen.class_indices.keys())) # to define number of classes in dense layer

        # create pre-trained model (you can built on pretrained model such as :  efficientnet, VGG , Resnet )
        # we will use efficientnetb3 from EfficientNet family.
        base_model = keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')

        model = keras.Sequential([
            base_model,
            keras.layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
            keras.layers.Dense(256, kernel_regularizer= keras.regularizers.l2(l2= 0.016), activity_regularizer= keras.regularizers.l1(0.006),
                        bias_regularizer= keras.regularizers.l1(0.006), activation= 'relu'),
            keras.layers.Dropout(rate= 0.45, seed= 123),
            keras.layers.Dense(4, activation= 'softmax')
        ])
        model.load_weights(self.checkpoint_weight)

        model.compile(keras.optimizers.Adamax(learning_rate=0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

        model.summary()

        return model

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

    # def loadPredictImage(self, picturePath):
    #     img_width, img_height = self.img_size
    #     test_image = image.load_img(picturePath, target_size=(img_width, img_height))
    #     test_image = image.img_to_array(test_image)
    #     test_image = np.expand_dims(test_image, axis=0)
    #     test_image = test_image.reshape(self.img_size[0], self.img_size[1], 3)
    #     print("image width: ", test_image.shape)
    #     # result = model.predict(test_image)
    #     return test_image
