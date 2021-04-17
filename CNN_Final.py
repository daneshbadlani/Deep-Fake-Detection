from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import gdown
from zipfile import ZipFile
import shutil
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Sequential, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Embedding,
    Dropout,
    SpatialDropout1D,
    Activation,
    Input,
    Reshape,
    BatchNormalization,
    ReLU,
    Conv1D,
    Flatten,
    AveragePooling1D,
    LeakyReLU,
)
from tensorflow.keras.losses import (BinaryCrossentropy)
from tensorflow.keras.optimizers import RMSprop
import os
from PIL import Image
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tqdm import tqdm
from IPython import display
import time


"""REAL IMAGE PREPROCESSING"""
PIC_DIR = './celeba/img_align_celeba/'
IMAGES_COUNT = 10000
ORIG_WIDTH = 178
ORIG_HEIGHT = 208
diff = (ORIG_HEIGHT - ORIG_WIDTH) // 2
WIDTH = 128
HEIGHT = 128
crop_rect = (0, diff, ORIG_WIDTH, ORIG_HEIGHT - diff)
realImages = []
for imageName in tqdm(os.listdir(PIC_DIR)[IMAGES_COUNT:20000]):
    pic = Image.open(PIC_DIR + imageName).crop(crop_rect)
    pic.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS)
    realImages.append(np.uint8(pic))  # Normalize the images
realImages = np.array(realImages) / 255
#print("Images Shape:", realImages.shape)
#plt.figure(1, figsize=(10, 10))

# Saving sample 25 images to see what faces look like
# for i in range(25):
#    plt.subplot(5, 5, i+1)
#    plt.imshow(realImages[i])
# plt.axis('off')
# plt.savefig('./test')

"""FAKE IMAGE PREPROCESSING"""
PIC_DIR = './fakeImages30000/'
IMAGES_COUNT = 10000
ORIG_WIDTH = 178
ORIG_HEIGHT = 208
diff = (ORIG_HEIGHT - ORIG_WIDTH) // 2
WIDTH = 128
HEIGHT = 128
crop_rect = (0, diff, ORIG_WIDTH, ORIG_HEIGHT - diff)
fakeImages = []
for imageName in tqdm(os.listdir(PIC_DIR)[:10000]):
    pic = Image.open(PIC_DIR + imageName)
    pic.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS)
    fakeImages.append(np.uint8(pic))  # Normalize the images
fakeImages = np.array(fakeImages) / 255
#print("Images Shape:", fakeImages.shape)
#plt.figure(1, figsize=(10, 10))

# Saving sample 25 images to see what faces look like
# for i in range(25):
#    plt.subplot(5, 5, i+1)
#    plt.imshow(fakeImages[i])
#    # plt.axis('off')
# plt.savefig('./test1')

# Load the dataset and split it
dataSet = np.concatenate([fakeImages, realImages], axis=0)
print("Dataset Shape:", dataSet.shape)
# y_values = []
y_values = np.concatenate([np.zeros(IMAGES_COUNT), np.ones(IMAGES_COUNT)])
print("Y SHAPE:", y_values.shape)
print("Sample:", y_values)
# for i in range(0, 10000):
#     y_values[i] = 0
# for i in range(10001, 20000):
#     y_values[i] = 1
X_train, X_test, y_train, y_test = train_test_split(dataSet, y_values,
                                                    shuffle=True, test_size=0.25)

# Print out shapes of training and testing sets
print("X Train:", X_train.shape)
print("X Test:", X_test.shape)
print("Y Train:", y_train.shape)
print("Y Test:", y_test.shape)


# Define the model
model = Sequential()

# Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, same padding and input shape
model.add(Conv2D(filters=32, input_shape=(128, 128, 3), kernel_size=(
    3, 3), activation='relu', kernel_initializer="he_uniform", padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))

# Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
                 kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))

# Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                 kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))

# Flatten the resulting data
model.add(Flatten())

# Add a dense layer with 128 nodes, relu activation and he uniform kernel initializer
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

# Add a batch normalization layer
model.add(BatchNormalization())

# Add dropout layer of 0.2
model.add(Dropout(rate=0.2))

# Add a dense softmax layer
model.add(Dense(1, activation='sigmoid'))

# Set up early stop training with a patience of 3
stop = EarlyStopping(patience=3)

# Compile the model with adam optimizer, binary cross entropy and accuracy metrics
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Fit the model with the generated data, 25 epochs, steps per epoch and validation data defined.
history = model.fit(X_train, y_train, epochs=2,
                    callbacks=[stop], validation_data=(X_test, y_test))

# plot accuracy vs epochs
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
plt.savefig('./accuracyPlot')

# plot loss vs epochs
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.savefig('./lossPlot')
