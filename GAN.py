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
)
from tensorflow.keras.callbacks import EarlyStopping
import librosa
import pathlib
import os
from IPython import display

from scipy import signal
from sklearn.model_selection import train_test_split

# from keras.datasets import spoken_digit
from scipy.io import wavfile

# from keras_adversarial import AdversarialModel, simple_gan, gan_targets
# from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling

# Data Load and Preprocess

files = os.listdir("./free-spoken-digit-dataset/recordings")
data = []
for i in files:
    x, sr = librosa.load("./free-spoken-digit-dataset/recordings/" + i)
    data.append(x)
print("Sampling Rate:", sr)
X = []
for i in range(len(data)):
    X.append(abs(librosa.stft(data[i]).mean(axis=1).T))
X = np.array(X)
y = []
for i in range(len(data)):
    y.append(1)
y = np.array(y)

# Splitting the data into 75% train and 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Outputting data shapes and samples
print("Data Shape:", X_train.shape)
print("Sample Record:", X_train[0])
print("Targets Shape:", y_train.shape)
print("Sample Target:", y_train[0])

# RNN Generator
generator_model = Sequential(
    [
        LSTM(200, input_shape=1025, dropout=0.2, recurrent_dropout=0.2),
        LSTM(200, dropout=0.2, recurrent_dropout=0.2),
        Dense(1025),
    ]
)

# FFNN Discriminator
discriminator_model = Sequential(
    [
        Input(shape=(1025,)),
        Dense(300, activation="relu"),
        Dense(300, activation="relu"),
        Dense(100, activation="sigmoid"),
    ]
)
"""
GAN_Network = simple_gan(
    generator_model, discriminator_model, normal_latent_sampling((100,))
)

GAN_Model = AdversarialModel(
    base_model=GAN_Network,
    player_params=[
        generator_model.trainable_weights,
        discriminator_model.trainable_weights,
    ],
)

GAN_Model.adversarial_compile(
    adversarial_optimizer=AdversarialOptimizerSimultaneous(),
    player_optimizer=["adam", "adam"],
    loss="binary_crossentropy",
)

generator_model.summary()
discriminator_model.summary()
GAN_Model.summary()
"""
