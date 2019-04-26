# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
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
n_epochs = 50
learning_rate = 0.001
dropout_rate = 0.3

# This variable is related to batch size.
# obtain this number by running test.py
steps_per_epoch = 20371

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
    optimizer=Adam(lr=learning_rate),
    loss={
        'duration_output': 'categorical_crossentropy',
        'rest_output': 'categorical_crossentropy'
    },
    loss_weights={
        'duration_output': 0.5,
        'rest_output': 0.5
    },
    metrics=['accuracy'])

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
