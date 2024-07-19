"""
Title: hurricane_damage_classification_CNN

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
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix

import matplotlib
# Graphical backend to matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import seaborn as sns

import urllib
import zipfile

import os
import sys

#############
tf.keras.backend.clear_session()
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Appends directory structure to PATH variable
current_dir = os.path.dirname(__file__)
common_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'common', 'utils'))
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
train_data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'train'))
valdn_data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'validation'))
sys.path.append(common_dir)

import tfToolkit
from tfToolkit import customMetricCallback
import tfLearningRatePlot

###############

NUM_EPOCHS = 10 # 10

def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.

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

    # Code to define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu,
                               input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    # Code to compile and train the model
    model.compile(optimizer=RMSprop(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
                  # YOUR CODE HERE
                  )

    history_fit=model.fit(train,
                   epochs=NUM_EPOCHS,
                   verbose=1,
                   validation_data=valid
    )

    print("\nCalling plot_train_val Plot Function..")
    tfToolkit.plot_train_val(history_fit)

    return model, valid

if __name__ == '__main__':
    model, valid = solution_model()

    # Generate predictions
    valid.reset()  # Reset the generator
    predictions = model.predict(valid, steps=valid.n // valid.batch_size + 1)
    predicted_classes = np.where(predictions > 0.5, 1, 0).flatten()

    # Get the true labels
    true_classes = valid.classes
    class_labels = list(valid.class_indices.keys())

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()



