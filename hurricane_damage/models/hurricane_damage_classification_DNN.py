"""
Title: hurricane_damage_classification_DNN

Description: Runs a classification of remotely sensed images taken to identify
  hurricane damage.

  Original Source: https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized
    The dataset consists of satellite images from Texas after Hurricane Harvey
    divided into two groups (damage and no_damage).

Usage: Run direct from IDE

Arguments: n/a Data location is hardcoded

Functions:

Classes:

Examples:

Author: Brendon Wolff-Piggott

Date: 19 July 2024
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import urllib
import zipfile

import os
import sys

#############
tf.keras.backend.clear_session()
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

###############

NUM_EPOCHS = 30

# Appends directory structure to PATH variable
current_dir = os.path.dirname(__file__)
common_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'common', 'utils'))
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
train_data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'train'))
valdn_data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'validation'))
sys.path.append(common_dir)

def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.
  #  download_and_extract_data()

    IMG_SIZE = 128
    BATCH_SIZE = 64

    # ImageDataGenerator is deprecated. Does not create tf datasets,
    #   but rather a python generator
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train = train_datagen.flow_from_directory(train_data_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
                                              class_mode="binary")
    valid = train_datagen.flow_from_directory(valdn_data_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
                                              class_mode="binary")

    # Simple DNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    from tensorflow.keras.optimizers import RMSprop

    # Code to compile and train the model
    model.compile(optimizer=RMSprop(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
                  # YOUR CODE HERE
                  )

    model.fit(train,
                   epochs=NUM_EPOCHS,
                   verbose=1,
                   validation_data=valid
    )

    return model

if __name__ == '__main__':
    model = solution_model()


