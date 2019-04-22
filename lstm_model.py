# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Internal import
from midi_data import MidiData
from utils import get_file_path, phrase_to_multihot, gen_batch

# import constants
from utils import DATASET_PATH, LENGTH_LIMIT
from utils import INTERVAL_RANGE, DURATION_RANGE, REST_RANGE

# Model parameters
n_steps = LENGTH_LIMIT
n_neurons = 100
n_inputs = INTERVAL_RANGE + DURATION_RANGE + REST_RANGE
n_outputs = DURATION_RANGE + REST_RANGE
batch_size = 10
n_epochs = 100
learning_rate = 0.001

# Data preparation
raw_Data = [[[i + j] for j in range(5)] for i in range(100)]
raw_target = [(i + 5) for i in range(100)]

data = np.array(raw_Data, dtype=float)
target = np.array(raw_target, dtype=float)

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
