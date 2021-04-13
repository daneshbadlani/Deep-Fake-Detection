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
from tensorflow.keras.losses import (
    BinaryCrossentropy
)
from tensorflow.keras.callbacks import EarlyStopping
import os
from PIL import Image
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tqdm import tqdm
from IPython import display
import time

"""IMAGE PREPROCESSING"""
PIC_DIR = './celeba/img_align_celeba/'
IMAGES_COUNT = 10000
ORIG_WIDTH = 178
ORIG_HEIGHT = 208
diff = (ORIG_HEIGHT - ORIG_WIDTH) // 2
WIDTH = 128
HEIGHT = 128
crop_rect = (0, diff, ORIG_WIDTH, ORIG_HEIGHT - diff)
images = []
for imageName in tqdm(os.listdir(PIC_DIR)[:IMAGES_COUNT]):
    pic = Image.open(PIC_DIR + imageName).crop(crop_rect)
    pic.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS)
    images.append(np.uint8(pic))  # Normalize the images
images = np.array(images) / 255
print("Images Shape:", images.shape)
plt.figure(1, figsize=(10, 10))

# Saving sample 25 images to see what faces look like
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(images[i])
    # plt.axis('off')
plt.savefig('./test')

"""CREATING MODELS"""
NUMSAMPLES = 16
LATENTDIM = 32
CHANNELS = 3

generator = Sequential([
    Input(shape=(LATENTDIM,)),

    Dense(128*16*16),
    LeakyReLU(),
    Reshape((16, 16, 128)),

    Conv2D(256, 5, padding='same'),
    LeakyReLU(),

    # try (2,2) for strides
    Conv2DTranspose(256, 4, strides=2, padding='same'),
    LeakyReLU(),

    # try (2,2) for strides
    Conv2DTranspose(256, 4, strides=2, padding='same'),
    LeakyReLU(),

    # try (2,2) for strides
    Conv2DTranspose(256, 4, strides=2, padding='same'),
    LeakyReLU(),

    Conv2D(512, 5, padding='same'),
    LeakyReLU(),

    Conv2D(512, 5, padding='same'),
    LeakyReLU(),

    Conv2D(CHANNELS, 7, activation='tanh', padding='same')
])

generator.summary()
# print("Generator Output Shape:", generator.output_shape)

discriminator = Sequential([
    Input(shape=(HEIGHT, WIDTH, CHANNELS)),

    Conv2D(256, 3),
    LeakyReLU(),

    # try (2,2) for strides
    Conv2D(256, 4, strides=2),
    LeakyReLU(),

    # try (2,2) for strides
    Conv2D(256, 4, strides=2),
    LeakyReLU(),

    # try (2,2) for strides
    Conv2D(256, 4, strides=2),
    LeakyReLU(),

    # try (2,2) for strides
    Conv2D(256, 4, strides=2),
    LeakyReLU(),

    Flatten(),
    Dropout(0.4),

    Dense(1, activation='sigmoid')
])

# Not sure why we'd do this. It freezes all weights and prevents updating them
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False

"""CREATE THE GAN"""
GAN = Sequential([
    Input(shape=(LATENTDIM,)),
    generator,
    discriminator
])

GAN.compile(loss='binary_crossentropy',
            optimizer='adam', metrics=['accuracy'])

# GAN.summary()


noise = np.random.normal(0, 1, (NUMSAMPLES, LATENTDIM))

"""SAVING NOISY FIGURES TO SEE WHAT THEY'D LOOK LIKE PRE-TRAINING"""
# imgs = generator.predict(noise)
# print("Discriminator Test:", discriminator.predict(imgs))

# fig = plt.figure(figsize=(40, 10))

# for i, img in enumerate(imgs[:4]):
#     ax = fig.add_subplot(1, NUMSAMPLES, i+1)
#     ax.imshow(img)
# fig.suptitle("Generated images", fontsize=30)
# plt.savefig('./noise')

"""TRAIN THE GAN"""
EPOCHS = 50  # 20000
batch_size = 16
RES_DIR = './generated'
FILE_PATH = '%s/generated_%d.png'
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)
CONTROL_SIZE_SQRT = 6
control_vectors = np.random.normal(size=(CONTROL_SIZE_SQRT**2, LATENTDIM)) / 2
start = 0
d_losses = []
a_losses = []
images_saved = 0
for step in range(EPOCHS):
    print("EPOCH-", step)
    start_time = time.time()
    latent_vectors = np.random.normal(size=(batch_size, LATENTDIM))
    generated = generator.predict(latent_vectors)

    real = images[start:start + batch_size]
    combined_images = np.concatenate([generated, real])

    labels = np.concatenate(
        [np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += .05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)
    d_losses.append(d_loss)

    latent_vectors = np.random.normal(size=(batch_size, LATENTDIM))
    misleading_targets = np.zeros((batch_size, 1))

    a_loss = GAN.train_on_batch(latent_vectors, misleading_targets)
    a_losses.append(a_loss)

    start += batch_size
    if start > images.shape[0] - batch_size:
        start = 0

    if step % 50 == 49:
        # GAN.save_weights('gan.h5')

        # print('%d/%d: d_loss: %.4f,  a_loss: %.4f.  (%.1f sec)' %
        #   (step + 1, EPOCHS, d_loss, a_loss, time.time() - start_time))
        print("d loss, a loss:", d_loss, a_loss)
        print("Time:", time.time() - start_time)

        control_image = np.zeros(
            (WIDTH * CONTROL_SIZE_SQRT, HEIGHT * CONTROL_SIZE_SQRT, CHANNELS))
        control_generated = generator.predict(control_vectors)
        for i in range(CONTROL_SIZE_SQRT ** 2):
            x_off = i % CONTROL_SIZE_SQRT
            y_off = i // CONTROL_SIZE_SQRT
            control_image[x_off * WIDTH:(x_off + 1) * WIDTH, y_off * HEIGHT:(
                y_off + 1) * HEIGHT, :] = control_generated[i, :, :, :]
        im = Image.fromarray(np.uint8(control_image * 255))
        im.save(FILE_PATH % (RES_DIR, images_saved))
        images_saved += 1
