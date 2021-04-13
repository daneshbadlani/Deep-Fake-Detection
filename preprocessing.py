import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from tensorflow.random import normal
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,Input,LeakyReLU,Conv2DTranspose,Reshape,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow import ones_like, zeros_like, GradientTape
from tensorflow.train import Checkpoint
from tensorflow.keras.preprocessing.text import Tokenizer
import imageio
import shutil
from PIL import Image
import time

ROOT = os.getcwd() #Establish root directory

# Create target directories for output, if they don't already exist
if not os.path.exists(ROOT + '/generated_images'):
    os.mkdir(ROOT + '/generated_images')
if not os.path.exists(ROOT + "/training_checkpoints"):
    os.mkdir(ROOT + "/training_checkpoints")

IMG_DIR = 'img_align_celeba/'

ORIG_WIDTH = 178
ORIG_HEIGHT = 208
diff = (ORIG_HEIGHT - ORIG_WIDTH) // 2
WIDTH = 128
HEIGHT = 128
BUFFER_SIZE = 60000
BATCH_SIZE = 256

crop_rect = (0, diff, ORIG_WIDTH, ORIG_HEIGHT - diff)

#Get image data
images = []
count = 0
print(os.getcwd())
for file in os.listdir(IMG_DIR): #For image in the directory,
    if count >= 100:
        break
    else: count += 1
    img = Image.open(IMG_DIR + file).crop(crop_rect) #open it and crop it
    img.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS) # get its pixel data
    images.append(np.uint8(img)) # append pixel data to training data list

#Normalize images
images = np.array(images) /255
print(images.shape)

#Generator
def make_generator_model():
    model = Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

#Discriminator
def make_discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                    input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model

#Create generator and discriminator to initialize model
generator = make_generator_model()

noise = normal([1, 100])
generated_image = generator(noise, training=False)

discriminator = make_discriminator_model()
decision = discriminator(generated_image)

#Show example image of noise
plt.figure(figsize=(12,12))
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()

#Set loss functions
cross_entropy = BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(ones_like(real_output), real_output)
    fake_loss = cross_entropy(zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(ones_like(fake_output), fake_output)

# Initialize generator and discriminator optimizer as Adam
# with learning rate = 0.0001
generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)

# Create a checkpoint directory 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = Checkpoint(generator_optimizer=generator_optimizer,
                          discriminator_optimizer=discriminator_optimizer,
                          generator=generator,
                          discriminator=discriminator)

# Define function for one training step to be called iteratively
def train_step(images):
    noise = normal([BATCH_SIZE, noise_dim])

    with GradientTape() as gen_tape, GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
       
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
       
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Plot and save prediction images as the GAN produces a new deepfake
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig(ROOT + '/generated_images/'+ 'image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()

# Main training loop for the GAN
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        display.clear_output(wait=True)
        generate_and_save_images(generator,epoch + 1,seed)
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    display.clear_output(wait=True)
    generate_and_save_images(generator,epochs,seed)

# Define training metrics and then train the model
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

seed = normal([num_examples_to_generate, noise_dim])

train(images, EPOCHS)



# ------------------- GIF Drawing --------------------
'''
Takes output images from the training period and creates a GIF that
shows how the GAN subtly made changes to create and accurate deep fake
'''

import imageio
import glob

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
