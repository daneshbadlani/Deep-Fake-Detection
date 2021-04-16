"""315: Final Assignment"""

from tensorflow.keras.models import (save_model, load_model)
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
from tensorflow.keras.losses import (
    BinaryCrossentropy
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
import os
from PIL import Image
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tqdm import tqdm
from IPython import display
import time

"""DOWNLOAD AND UNZIP IMAGE DATASET"""

# os.makedirs("celeba_gan")
#if not os.path.exists('./celeba'):
    #url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
    #output = "celeba/data.zip"
    #gdown.download(url, output, quiet=True)

    #with ZipFile("celeba/data.zip", "r") as zipobj:
      #  zipobj.extractall("celeba")
   # print("DATA DOWNLOADED AND UNZIPPED!")

"""IMAGE PREPROCESSING"""
#PIC_DIR = './celeba/img_align_celeba/'
#IMAGES_COUNT = 10000
#ORIG_WIDTH = 178
#ORIG_HEIGHT = 208
#diff = (ORIG_HEIGHT - ORIG_WIDTH) // 2
#WIDTH = 128
#HEIGHT = 128
#crop_rect = (0, diff, ORIG_WIDTH, ORIG_HEIGHT - diff)
#images = []
#for imageName in tqdm(os.listdir(PIC_DIR)[:IMAGES_COUNT]):
#    pic = Image.open(PIC_DIR + imageName).crop(crop_rect)
#    pic.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS)
#    images.append(np.uint8(pic))  # Normalize the images
#images = np.array(images) / 255
#print("Images Shape:", images.shape)
#plt.figure(1, figsize=(10, 10))

# Saving sample 25 real images to see what faces look like
#for i in range(25):
#    plt.subplot(5, 5, i+1)
#    plt.imshow(images[i])
    # plt.axis('off')
#plt.savefig('./test')

"""CREATING MODELS"""



"""CREATE THE GAN"""


"""FUNCTIONS TO SAVE AND LOAD MODELS"""




def load():
    discriminator = load_model('./saved40000/'+ 'discriminator')
    generator = load_model('./saved40000/' + 'generator')
    gan = load_model('./saved40000/'+ 'gan')
    gan.summary()
    # discriminator.summary()
    # generator.summary()

    return gan, generator, discriminator

GAN, generator, discriminator = load()
print("MODELS LOADED!")
"""TRAIN THE GAN [ONLY IF SAVED MODELS NOT FOUND IN ./saved]"""
EPOCHS = 10000  # 20000
BATCH_SIZE = 16
RES_DIR = './generated'
FILE_PATH = '%s/generated_%d.png'
save_dir = './saved40000/'

LATENTDIM = 32
"""SAVING GENERATED IMAGES FOR NEXT MODEL"""
noise = np.random.normal(0, 1, (10000, LATENTDIM))
imgs = generator.predict(noise)
# print("Discriminator Test:", discriminator.predict(imgs))

if not os.path.exists('./fakeImages'):
    os.makedirs('./fakeImages')
for i, img in enumerate(imgs):
    image = Image.fromarray(np.uint8(img * 255))
    image.save('./fakeImages/fake_%d.png' % i)
