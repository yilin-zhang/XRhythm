# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam

# Internal imports
from utils import gen_batch

# import constants
from utils import DATASET_PATH, LENGTH_LIMIT
from utils import INTERVAL_RANGE, DURATION_RANGE, REST_RANGE

# Fixed model parameters
n_steps = LENGTH_LIMIT
n_inputs = INTERVAL_RANGE + DURATION_RANGE + REST_RANGE
n_outputs = INTERVAL_RANGE + DURATION_RANGE + REST_RANGE

# Changable model parameters
n_neurons = 64
batch_size = 10
n_epochs = 100
learning_rate = 0.001

# Construct Model
model = Sequential()

model.add(
    Dense(
        n_neurons,
        batch_input_shape=(batch_size, n_steps, n_inputs),
        activation='relu'))

model.add(
    LSTM(
        n_neurons,
        batch_input_shape=(batch_size, n_steps, n_neurons),
        activation='relu',
        return_sequences=True))

model.add(
    LSTM(
        n_neurons,
        batch_input_shape=(batch_size, n_steps, n_neurons),
        activation='relu',
        return_sequences=True))

model.add(Dense(n_outputs, activation='softmax'))

model.compile(
    optimizer=Adam(lr=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.summary()
# Fit model
gen = gen_batch('./dataset/train', n_steps, batch_size)
model.fit_generator(gen, steps_per_epoch=8000, epochs=n_epochs)

# # model.summary()
# for step in range(n_epochs):
#     X_batch, y_batch = next(gen)
#     cost = model.train_on_batch(X_batch, y_batch)
#     print(cost)
