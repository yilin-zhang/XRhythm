# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
# from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout
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
n_outputs = DURATION_RANGE + REST_RANGE

# Changeable model parameters (hyper parameters)
n_neurons = 256
batch_size = 40
# n_epochs = 100
n_epochs = 1
learning_rate = 0.001
dropout_rate = 0.3

# This variable is related to batch size.
# obtain this number by running test.py
# steps_per_epoch = 20371
steps_per_epoch = 10

# Set dataset path
train_path = DATASET_PATH + '/train'
valid_path = DATASET_PATH + '/valid'

# # Construct Model
model_input = Input(shape=(n_steps, n_inputs))
x = LSTM(n_neurons, activation='relu', return_sequences=True)(model_input)
x = Dropout(dropout_rate)(x)
x = LSTM(n_neurons, activation='relu', return_sequences=True)(x)
x = Dropout(dropout_rate)(x)
duration_output = Dense(
    DURATION_RANGE, activation='softmax', name='duration_output')(x)
rest_output = Dense(REST_RANGE, activation='softmax', name='rest_output')(x)
model = Model(inputs=model_input, outputs=[duration_output, rest_output])

model.compile(
    optimizer='Adam',
    loss={
        'duration_output': 'categorical_crossentropy',
        'rest_output': 'categorical_crossentropy'
    },
    loss_weights={
        'duration_output': 0.5,
        'rest_output': 0.5
    },
    metrics=['accuracy'])

# model = Sequential()
# # model.add(Dense(n_neurons, input_shape=(n_steps, n_inputs), activation='relu'))
# model.add(
#     LSTM(
#         n_neurons,
#         input_shape=(n_steps, n_inputs),
#         activation='relu',
#         return_sequences=True))
# model.add(Dropout(dropout_rate))
# model.add(
#     LSTM(
#         n_neurons,
#         input_shape=(n_steps, n_inputs),
#         activation='relu',
#         return_sequences=True))
# model.add(Dropout(dropout_rate))
# model.add(Dense(n_outputs, activation='softmax'))

# model.compile(
#     optimizer=Adam(lr=learning_rate),
#     loss='categorical_crossentropy',
#     metrics=['categorical_accuracy'])

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
