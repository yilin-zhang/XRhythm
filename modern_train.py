# -*-coding: utf-8-*-
# Author: Yilin Zhang

from model import XRModel
from configs import DATASET_PATH

# Changeable parameters
batch_size = 40
dropout_rate = 0.3
n_epochs = 50

# These variables are related to batch size.
# Obtain these two numbers by running cal_batches.py
# in the function gen_batch (the definition is in utils.py).
steps_per_epoch = 21132
validation_steps = 4495

# Set dataset path
train_path = DATASET_PATH + '/train'
valid_path = DATASET_PATH + '/valid'

xrmodel = XRModel()
xrmodel.construct(dropout_rate)
xrmodel.compile()
xrmodel.fit(train_path, valid_path, batch_size, steps_per_epoch,
            validation_steps, n_epochs)
