import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class EyeDiseaseDataset:
    def __init__(self, dataDir):
        self.data_dir = dataDir

    def dataPaths(self):
        filepaths = []
        labels = []
        folds = os.listdir(self.data_dir)
        for fold in folds:
            foldPath = os.path.join(self.data_dir, fold)
            filelist = os.listdir(foldPath)
            for file in filelist:
                fpath = os.path.join(foldPath, file)
                filepaths.append(fpath)
                labels.append(fold)
        return filepaths, labels

    def dataFrame(self, files, labels):

        Fseries = pd.Series(files, name='filepaths')
        Lseries = pd.Series(labels, name='labels')
        return pd.concat([Fseries, Lseries], axis=1)

    def split_(self):
        files, labels = self.dataPaths()
        df = self.dataFrame(files, labels)
        strat = df['labels']
        trainData, dummyData = train_test_split(df, train_size=0.8, shuffle=True, random_state=42, stratify=strat)
        strat = dummyData['labels']
        validData, testData = train_test_split(dummyData, train_size=0.5, shuffle=True, random_state=42, stratify=strat)
        return trainData, validData, testData

    def augment_data(self, train_df, valid_df, test_df, batch_size=16):
        img_size = (256, 256)
        channels = 3
        color = 'rgb'

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.5, 1.5])

        valid_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

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
