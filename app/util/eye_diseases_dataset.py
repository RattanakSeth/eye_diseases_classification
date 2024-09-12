import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
class EyeDiseaseDataset:
    def __init__(self, dataDir):
        self.data_dir = dataDir
        # print("path: ", self.data_dir)

    def define_paths(self):
        filepaths = []
        labels = []

        folds = os.listdir(self.data_dir)
        for fold in folds:
            foldpath = os.path.join(self.data_dir, fold)
            # print("foldpath", foldpath)
            filelist = os.listdir(foldpath)
            for file in filelist:
                fpath = os.path.join(foldpath, file)
                filepaths.append(fpath)
                labels.append(fold)

        return filepaths, labels

    # def dataFrame(self, files, labels):
    #
    #     Fseries = pd.Series(files, name='filepaths')
    #     Lseries = pd.Series(labels, name='labels')
    #     return pd.concat([Fseries, Lseries], axis=1)

    def define_df(self, files, classes):
        # print("class: ", classes)
        Fseries = pd.Series(files, name='filepaths')
        Lseries = pd.Series(classes, name='labels')
        return pd.concat([Fseries, Lseries], axis=1)
    
    def split_data(self):
        # train dataframe
        files, classes = self.define_paths()
        # print("file: ", files)
        # print("classes: ", classes)
        df = self.define_df(files, classes)
        strat = df['labels']
        print("df: ", df)
        train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123, stratify=strat)

        # valid and test dataframe
        strat = dummy_df['labels']
        valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=strat)

        # print("data frame: ", test_df)

        return train_df, valid_df, test_df

    def augment_data(self, train_df, valid_df, test_df, batch_size=16):
        img_size = (256, 256)
        channels = 3
        color = 'rgb'

        train_datagen = ImageDataGenerator(
            rotation_range=30,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.5, 1.5])

        valid_test_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='filepaths',
            y_col='labels',
            target_size=img_size,
            color_mode=color,
            batch_size=batch_size,
            shuffle=True,
            class_mode='categorical'
        )

        print("Shape of augmented training images:", train_generator.image_shape)

        valid_generator = valid_test_datagen.flow_from_dataframe(
            valid_df,
            x_col='filepaths',
            y_col='labels',
            target_size=img_size,
            color_mode=color,
            batch_size=batch_size,
            shuffle=True,
            class_mode='categorical'
        )

        print("Shape of validation images:", valid_generator.image_shape)

        test_generator = valid_test_datagen.flow_from_dataframe(
            test_df,
            x_col='filepaths',
            y_col='labels',
            target_size=img_size,
            color_mode=color,
            batch_size=batch_size,
            shuffle=False,
            class_mode='categorical'
        )

        print("Shape of test images:", test_generator.image_shape)

        return train_generator, valid_generator, test_generator

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
    
    def split_data_sample_input(self):
        # train dataframe
        files, classes = self.define_paths()
        # print("file: ", files)
        # print("classes: ", classes)
        df = self.define_df(files, classes)
        strat = df['labels']
        print("df: ", df)
        train_df, dummy_df = train_test_split(df) #train_size=0.8, shuffle=True, random_state=2, stratify=strat
        print("train_df: ", train_df)
        print("dummy_df: ", dummy_df)
        # valid and test dataframe
        strat = dummy_df['labels']
        valid_df, test_df = train_test_split(dummy_df) # train_size=0.5, shuffle=True, random_state=2, stratify=strat

        print("valid_df: ", valid_df)
        print("test df: ", test_df)

        return train_df, valid_df, test_df
