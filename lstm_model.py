# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, LeakyReLU
from keras.optimizers import Adadelta
from keras.callbacks import TensorBoard, ModelCheckpoint

# Internal imports
from utils import gen_batch

# import constants
from configs import DATASET_PATH, LENGTH_LIMIT
from configs import INTERVAL_RANGE, DURATION_RANGE, REST_RANGE

# Fixed model parameters
n_steps = LENGTH_LIMIT
n_inputs = INTERVAL_RANGE + DURATION_RANGE + REST_RANGE
n_outputs = DURATION_RANGE + REST_RANGE

# Changeable model parameters (hyper parameters)
n_neurons = 256
batch_size = 40
n_epochs = 50
dropout_rate = 0.3

# These variables are related to batch size.
# Obtain these two numbers by running cal_batches.py
# in the function gen_batch (the definition is in utils.py).
steps_per_epoch = 21132
validation_steps = 4495

# Set dataset path
train_path = DATASET_PATH + '/train'
valid_path = DATASET_PATH + '/valid'

# Construct Model
model_input = Input(batch_shape=(None, n_steps, n_inputs))
x = LSTM(n_neurons, activation='linear', return_sequences=True)(model_input)
x = LeakyReLU()(x)
x = Dropout(dropout_rate)(x)
x = LSTM(n_neurons, activation='linear', return_sequences=True)(x)
x = LeakyReLU()(x)
x = Dropout(dropout_rate)(x)
x = LSTM(n_neurons, activation='linear', return_sequences=True)(x)
x = LeakyReLU()(x)
x = Dropout(dropout_rate)(x)
x = LSTM(n_neurons, activation='linear', return_sequences=True)(x)
x = LeakyReLU()(x)
x = Dropout(dropout_rate)(x)
duration_output = Dense(
    DURATION_RANGE, activation='softmax', name='duration_output')(x)
rest_output = Dense(REST_RANGE, activation='softmax', name='rest_output')(x)
model = Model(inputs=model_input, outputs=[duration_output, rest_output])

model.compile(
    optimizer=Adadelta(),
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
model_save_path = "./models/saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
mc_callback = ModelCheckpoint(filepath=model_save_path, monitor='val_loss')

# Summary the model
model.summary()

# Fit model
gen_train = gen_batch(train_path, n_steps, batch_size)
gen_valid = gen_batch(valid_path, n_steps, batch_size)
model.fit_generator(
    gen_train,
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs,
    validation_data=gen_valid,
    validation_steps=validation_steps,
    callbacks=[tb_callback, mc_callback],
    workers=2,
    use_multiprocessing=True)

# model.save('lstm_model.h5')
