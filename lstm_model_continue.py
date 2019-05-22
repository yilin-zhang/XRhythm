# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Dropout, LeakyReLU
from keras.optimizers import Adadelta
from keras.callbacks import TensorBoard, ModelCheckpoint

# Internal imports
from utils import gen_batch

# import constants
from configs import XADRUM_DATASET_PATH, LENGTH_LIMIT
from configs import INTERVAL_RANGE, DURATION_RANGE, REST_RANGE

# Fixed model parameters
n_steps = LENGTH_LIMIT
n_inputs = INTERVAL_RANGE + DURATION_RANGE + REST_RANGE
n_outputs = DURATION_RANGE + REST_RANGE

# Changeable model parameters (hyper parameters)
n_neurons = 256
batch_size = 5
n_epochs = 100
dropout_rate = 0.5
initial_epoch = 50

# These variables are related to batch size.
# Obtain these two numbers by running cal_batches.py
# Note that you MUST remove the outermost loop
# in the function gen_batch (the definition is in utils.py).
steps_per_epoch = 202
validation_steps = 11

# Set dataset path
train_path = XADRUM_DATASET_PATH + '/train'
valid_path = XADRUM_DATASET_PATH + '/valid'

MODEL_PATH = './models/201905030320/lstm_model.h5'
model = load_model(MODEL_PATH)

# Construct New Model
model_input = Input(shape=(n_steps, n_inputs))
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

new_model = Model(inputs=model_input, outputs=[duration_output, rest_output])

# Copy all the weights to the new model
for new_layer, layer in zip(new_model.layers[1:], model.layers[1:]):
    new_layer.set_weights(layer.get_weights())

new_model.compile(
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
gen_train = gen_batch(train_path, n_steps, batch_size, overlap=True)
gen_valid = gen_batch(valid_path, n_steps, batch_size, overlap=True)
model.fit_generator(
    gen_train,
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs,
    validation_data=gen_valid,
    validation_steps=validation_steps,
    callbacks=[tb_callback, mc_callback],
    workers=2,
    use_multiprocessing=True,
    initial_epoch=initial_epoch)

# model.save('lstm_model.h5')
