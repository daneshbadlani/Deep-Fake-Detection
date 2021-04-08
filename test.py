"""
Author: Ana Estrada
This basically will turn the wave files into spectrogram's that are ready to use. I'll put the like i used below.
I just think we need to change the dataset.

tutorial
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
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
#from keras.datasets import spoken_digit
from scipy.io import wavfile


"""Easier way to preprocess the audio requires the download of the files
https://medium.com/@nitinsingh1789/spoken-digit-classification-b22d67fd24b0
this will take all the files I think in any format and makes them into stft"""
#needs to be the path to the file of recording
file = os.listdir('free-spoken-digit-dataset/recordings')
data=[]
#needs to go through the 
for i in file:
    x , sr = librosa.load('free-spoken-digit-dataset/recordings/'+i)
    data.append(x)
#X is were the spectrograms are stored
X=[]
for i in range(len(data)):
    X.append(abs(librosa.stft(data[i]).mean(axis = 1).T))
X= np.array(X)
y= []
#assigning the label 1 for real audio
#use zero for fake audio
for i in range(len(data)):
    y.append(1)
y = mp.array(y)
X.to_csv("spectrogram",index = False)
