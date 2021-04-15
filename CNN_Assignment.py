"""
Author: Ana Estrada and Danesh Badlani
File: CNN_Assignment.py
"""


from keras.utils.np_utils import to_categorical   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

## Load is Cifar 10 dataset
(X_train, y_train), (X_test, y_test) = #IMAGE DATA

## Print out shapes of training and testing sets
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

## Normalize x train and x test images
#transfer learning maybe 
##X_train, X_test = X_train / 255.0, X_test / 255.0
##
##init_model = tf.keras.applications.#name of the transfer learing model(input_shape + IMG_SHAPE, include_top = FALSE, weights = 'imagenet')
##
##feature_batch = init_model(image_batch)
##
##init_model.trainable = False

## Define the model
model = Sequential()

## Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, same padding and input shape
model.add(Conv2D(filters = 32, input_shape = (218, 178, 3), kernel_size = (3, 3), activation = 'relu', kernel_initializer = "he_uniform", padding = 'same'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding 
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D(pool_size = (2,2)))

## Add dropout layer of 0.2
model.add(Dropout(rate = 0.2))

## Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding= 'same'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding= 'same'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D(pool_size = (2,2)))

## Add dropout layer of 0.2
model.add(Dropout(rate = 0.2))

## Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding= 'same'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D(pool_size = (2,2)))

## Add dropout layer of 0.2
model.add(Dropout(rate = 0.2))

## Flatten the resulting data
model.add(Flatten())

## Add a dense layer with 128 nodes, relu activation and he uniform kernel initializer
model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add dropout layer of 0.2
model.add(Dropout(rate = 0.2))

## Add a dense softmax layer
model.add(Dense(2, activation = 'softmax'))

## Set up early stop training with a patience of 3
stop = EarlyStopping(patience = 3, delta = 0.00001)

## Compile the model with adam optimizer, categorical cross entropy and accuracy metrics
model.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['accuracy'])


## Fit the model with the generated data, 200 epochs, steps per epoch and validation data defined. 
history = model.fit(X_train, y_train, epochs = 25,  callbacks = stop, validation_split = 0.1)

results = model.evaluate(X_test, y_test)

# plot accuracy vs epochs
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# plot loss vs epochs
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
