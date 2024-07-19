 # ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative to its
# difficulty. So your Category 1 question will score significantly less than
# your Category 5 question.
#
# WARNING: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure. You do not need them to solve the question.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# COMPUTER VISION WITH CNNs
#
# Create and train a classifier to classify images between two classes
# (damage and no_damage) using the satellite-images-of-hurricane-damage dataset.
# ======================================================================
#
# ABOUT THE DATASET
#
# Original Source:
# https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized
# The dataset consists of satellite images from Texas after Hurricane Harvey
# divided into two groups (damage and no_damage).
# ==============================================================================
#
# INSTRUCTIONS
#
# We have already divided the data for training and validation.
#
# Complete the code in following functions:
# 1. preprocess()
# 2. solution_model()
#
# Your code will fail to be graded if the following criteria are not met:
# 1. The input shape of your model must be (128,128,3), because the testing
#    infrastructure expects inputs according to this specification. You must
#    resize all the images in the dataset to this size while pre-processing
#    the dataset.
# 2. The last layer of your model must be a Dense layer with 1 neuron
#    activated by sigmoid since this dataset has 2 classes.
#
# HINT: Your neural network must have a validation accuracy of approximately
# 0.95 or above on the normalized validation dataset for top marks.

import urllib
import zipfile

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from Pillow import PIL

# This function downloads and extracts the dataset to the directory that
# contains this file.
# DO NOT CHANGE THIS CODE
# (unless you need to change https to http)
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/satellitehurricaneimages.zip'
    urllib.request.urlretrieve(url, 'satellitehurricaneimages.zip')
    with zipfile.ZipFile('satellitehurricaneimages.zip', 'r') as zip_ref:
        zip_ref.extractall()

def preprocess(image, label):
    # NORMALIZE YOUR IMAGES HERE (HINT: Rescale      by 1/.255)
    ## COMMENT 15 Feb 2024: This makes no sense - you don't normalise
    ##    the image labels - which would be 0 or 1

    image = tf.cast(image, tf.float32) / 255  # Ensure image is float32 and normalize
    label = tf.cast(label, tf.float32) / 255  # Cast label to float32 before division

    return image, label


def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.
    download_and_extract_data()

    IMG_SIZE = 128
    BATCH_SIZE = 64

    # ImageDataGenerator is deprecated. Does not create tf datasets,
    #   but rather a python generator
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)
    train = train_datagen.flow_from_directory("train", target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
                                              class_mode="binary")
    valid = train_datagen.flow_from_directory("validation", target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
                                              class_mode="binary")

    # Code to define the model
    model = tf.keras.models.Sequential([
        # ADD LAYERS OF THE MODEL HERE
#        tf.keras.layers.Flatten(),
#        tf.keras.layers.Dense(64),
        # If you don't adhere to the instructions in the following comments,
        # tests will fail to grade your model:
        # The input layer of your model must have an input shape of
        # (128,128,3).
        # Make sure your last layer has 1 neuron activated by sigmoid.
        tf.keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu,
                               input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    from tensorflow.keras.optimizers import RMSprop

    # Code to compile and train the model
    model.compile(optimizer=RMSprop(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
                  # YOUR CODE HERE
                  )

# Removed suffix _ds from datasets below, because they are no longer TF datasets
    model.fit(train,
                   epochs=15,
                   verbose=1,
                   validation_data=valid

        # YOUR CODE HERE
    )

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

