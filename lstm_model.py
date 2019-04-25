# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

# Internal imports
from utils import gen_batch

# import constants
from utils import DATASET_PATH, LENGTH_LIMIT
from utils import INTERVAL_RANGE, DURATION_RANGE, REST_RANGE

# Fixed model parameters
n_steps = LENGTH_LIMIT
n_inputs = INTERVAL_RANGE + DURATION_RANGE + REST_RANGE
n_outputs = INTERVAL_RANGE + DURATION_RANGE + REST_RANGE

# Changeable model parameters (hyper parameters)
n_lstm_layers = 2
n_neurons = 256
batch_size = 10
n_epochs = 100
learning_rate = 0.001
use_dropout = True

# This variable is related to batch size.
# obtain this number by running test.py
steps_per_epoch = 81487

# Set dataset path
train_path = DATASET_PATH + '/train'
valid_path = DATASET_PATH + '/valid'

# Construct Model
model = Sequential()
model.add(Dense(n_neurons, input_shape=(n_steps, n_inputs), activation='relu'))
for _ in range(n_lstm_layers):
    model.add(LSTM(n_neurons, activation='relu', return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(Dense(n_outputs, activation='softmax'))

model.compile(
    optimizer=Adam(lr=learning_rate),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'])

# Set tensorboard callback
tb_callback = TensorBoard(log_dir='./logs', batch_size=batch_size)

# Summary the model
model.summary()

# Fit model
gen_train = gen_batch(train_path, n_steps, batch_size)
model.fit_generator(
    gen_train,
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs,
    callbacks=[tb_callback])

model.save('lstm_model.h5')

# for step in range(n_epochs):
#     X_batch, y_batch = next(gen)
#     cost = model.train_on_batch(X_batch, y_batch)
#     print(cost)
