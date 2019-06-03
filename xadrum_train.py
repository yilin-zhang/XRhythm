# -*-coding: utf-8-*-
# Author: Yilin Zhang

from model import XRModel
from configs import XADRUM_DATASET_PATH, MODEL_PATH

# Changeable parameters
batch_size = 5
dropout_rate = 0.6
n_epochs = 100
initial_epoch = 50

# These variables are related to batch size.
# Obtain these two numbers by running cal_batches.py
# in the function gen_batch (the definition is in utils.py).
steps_per_epoch = 202
validation_steps = 11

# Select the model.
model_path = MODEL_PATH + '/201905310621/saved-model-50-0.98.hdf5'

# Set dataset path
train_path = XADRUM_DATASET_PATH + '/train'
valid_path = XADRUM_DATASET_PATH + '/valid'

xrmodel = XRModel()
xrmodel.load(model_path, dropout_rate)
xrmodel.compile()
xrmodel.fit(train_path, valid_path, batch_size, steps_per_epoch,
            validation_steps, n_epochs, initial_epoch)
