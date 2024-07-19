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

# This function normalizes the images.
# COMPLETE THE CODE IN THIS FUNCTION
def preprocess(image, label):
    # NORMALIZE YOUR IMAGES HERE (HINT: Rescale      by 1/.255)

    image = tf.cast(image, tf.float32) / 255  # Ensure image is float32 and normalize
    label = tf.cast(label, tf.float32) / 255  # Cast label to float32 before division

    return image, label


# This function loads the data, normalizes and resizes the images, splits it into
# train and validation sets, defines the model, compiles it and finally
# trains the model. The trained model is returned from this function.

# COMPLETE THE CODE IN THIS FUNCTION.
def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.
    download_and_extract_data()

    IMG_SIZE = 128
    BATCH_SIZE = 64

    # The following code reads the training and validation data from their
    # respective directories, resizes them into the specified image size
    # and splits them into batches. You must fill in the image_size
    # argument for both training and validation data.
    # HINT: Image size is a tuple
    # B 7 Feb 2024 Note: The variable "image" referenced in tf.image.resize
    #   is not yet defined - throws error
    # image = tf.image.resize(image, size=[image_size, image_size])

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='train/', image_size= (IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='validation/', image_size= (IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE)

# Brendon 12 Feb 2024
#    Output from is a dataset object - see filed code on viewing shape / contents
#    data_shape = train_ds.shape
#    print("The data has shape", data_shape)
#    print("There are ", train_ds[0]," training examples with shape ",
#            (data_shape[1]), (data_shape[2]))

    # Normalizes train and validation datasets using the
    # preprocess() function.
    # Also makes other calls, as evident from the code, to prepare them for
    # training.
    # Do not batch or resize the images in the dataset here since it's already
    # been done previously.
    train_ds = train_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Code to define the model
    model = tf.keras.models.Sequential([
        # ADD LAYERS OF THE MODEL HERE
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),
        # If you don't adhere to the instructions in the following comments,
        # tests will fail to grade your model:
        # The input layer of your model must have an input shape of
        # (128,128,3).
        # Make sure your last layer has 1 neuron activated by sigmoid.
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    from tensorflow.keras.optimizers import RMSprop

    # Code to compile and train the model
    model.compile(optimizer=RMSprop(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
                  # YOUR CODE HERE
                  )

    model.fit(train_ds,
                   epochs=15,
                   verbose=1,
                   validation_data=val_ds

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

