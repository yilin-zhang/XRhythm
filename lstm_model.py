# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Internal import
from midi_data import MidiData
from utils import get_file_path

# import constants
from utils import DATASET_PATH, LENGTH_LIMIT


# TODO The inner for loop is not finished
# What is the batch size?
# How many steps? LENGTH_LIMIT?
# Import dataset
def get_data(dataset_path):
    for data_path, data_file in get_file_path(dataset_path, '.pkl'):
        with open(data_path, 'rb') as f:
            phrase_data = pickle.load(f)
        for phrase in phrase_data:
            X = phrase[:LENGTH_LIMIT]
            y = phrase[1:LENGTH_LIMIT + 1]
            yield X, y


# Convert note lists to multi-hot arrays

# Data preparation
raw_Data = [[[i + j] for j in range(5)] for i in range(100)]
raw_target = [(i + 5) for i in range(100)]

data = np.array(raw_Data, dtype=float)
target = np.array(raw_target, dtype=float)

data.shape
target.shape

x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=4)

# RNN Model
model = Sequential()
model.add(LSTM((1), batch_input_shape=(None, 5, 1), return_sequences=False))
model.compile(
    loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(
    x_train, y_train, epochs=50, validation_data=(x_test, y_test))

results = model.predict(x_test)
