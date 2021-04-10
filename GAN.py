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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import librosa
import pathlib
import os
from IPython import display

from scipy import signal
from sklearn.model_selection import train_test_split

# from keras.datasets import spoken_digit
from scipy.io import wavfile
import time

# from keras_adversarial import AdversarialModel, simple_gan, gan_targets
# from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling

"""Easier way to preprocess the audio requires the download of the files
https://medium.com/@nitinsingh1789/spoken-digit-classification-b22d67fd24b0
this will take all the files I think in any format and makes them into stft"""

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print("Data Shape:", X_train.shape)
print("Sample Record:", X_train[0])
print("Targets Shape:", y_train.shape)
print("Sample Target:", y_train[0])

"""
The MNIST GAN tutorial below has been modified to work with audio.
https://www.tensorflow.org/tutorials/generative/dcgan"""


def make_generator_model():
    model = Sequential()
    model.add(Dense(205, input_shape=(1025,)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Reshape((205, 1)))

    model.add(Conv1D(16, 20, padding="same"))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(32, 25, padding="same"))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(64, 50, padding="same"))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(5, 100, padding="same"))
    model.add(Dropout(rate=0.3))
    model.add(Flatten())
    return model


def make_discriminator_model():
    return Sequential(
        [
            Input(shape=(1025,)),
            Dense(300, activation="relu"),
            Dense(300, activation="relu"),
            Dense(100, activation="sigmoid"),
        ]
    )


# Loss function for both
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Compares dicriminator's predictions on real audio to array of 1s and
# fake audio to an array of 0s


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Optimizers for the two models
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 50
noise_dim = 1025
BATCH_SIZE = 250

# Batch and shuffle the training data
train_dataset = (
    tf.data.Dataset.from_tensor_slices(X_train)
    .shuffle(X_train.shape[0])
    .batch(BATCH_SIZE)
)

# Create the models
generator = make_generator_model()
generator.summary()
discriminator = make_discriminator_model()
discriminator.summary()

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".


@tf.function
def train_step(audios):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_audio = generator(noise, training=True)

        real_output = discriminator(audios, training=True)
        fake_output = discriminator(generated_audio, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for audio_batch in dataset:
            train_step(audio_batch)

        print("Time for epoch {} is {} sec".format(
            epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)


train(train_dataset, EPOCHS)
