# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
from keras.models import load_model
from keras.callbacks import TensorBoard

# Internal imports
from utils import gen_batch

# import constants
from utils import XADRUM_DATASET_PATH, LENGTH_LIMIT
from utils import INTERVAL_RANGE, DURATION_RANGE, REST_RANGE

# Fixed model parameters
n_steps = LENGTH_LIMIT
n_inputs = INTERVAL_RANGE + DURATION_RANGE + REST_RANGE
n_outputs = DURATION_RANGE + REST_RANGE

# Changeable model parameters (hyper parameters)
batch_size = 40
n_epochs = 100
initial_epoch = 50

# These variables are related to batch size.
# Obtain these two numbers by running test.py
# Note that you MUST remove the outermost loop
# in the function gen_batch (the definition is in utils.py).
steps_per_epoch = 21132
validation_steps = 4495

# Set dataset path
train_path = XADRUM_DATASET_PATH + '/train'
valid_path = XADRUM_DATASET_PATH + '/valid'

MODEL_PATH = './models/201905030320/lstm_model.h5'
model = load_model(MODEL_PATH)

# Set tensorboard callback
tb_callback = TensorBoard(log_dir='./logs', batch_size=batch_size)

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
    callbacks=[tb_callback],
    workers=2,
    use_multiprocessing=True,
    initial_epoch=initial_epoch)

model.save('lstm_model.h5')
