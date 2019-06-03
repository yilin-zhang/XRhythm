# -*-coding: utf-8-*-
# Author: Yilin Zhang

# Calculate the batch numbers, to obtain a proper steps_per_epoch
from utils import gen_batch
from configs import LENGTH_LIMIT, DATASET_PATH, XADRUM_DATASET_PATH

# Choose which dataset you want to check
#dataset_path = DATASET_PATH
dataset_path = XADRUM_DATASET_PATH

# Confirm your batch size before running this script.
batch_size = 5

n_steps = LENGTH_LIMIT
gen_train = gen_batch(dataset_path + '/train', n_steps, batch_size, loop=False)
gen_valid = gen_batch(dataset_path + '/valid', n_steps, batch_size, loop=False)
gen_test = gen_batch(dataset_path + '/test', n_steps, batch_size, loop=False)

n_train = 0
n_valid = 0
n_test = 0
for X_batch, y_batch in gen_train:
    n_train += 1
for _, _ in gen_valid:
    n_valid += 1
for _, _ in gen_test:
    n_test += 1

print('n_train:', n_train)
print('n_valid:', n_valid)
print('n_test:', n_test)
