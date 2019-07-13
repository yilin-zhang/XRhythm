# -*-coding: utf-8-*-
# Author: Yilin Zhang

import os
import copy
import numpy as np
import pickle
import random

from xrlib.configs import INTERVAL_THRESHOLD, DURATION_THRESHOLD, REST_THRESHOLD
from xrlib.configs import INTERVAL_RANGE, DURATION_RANGE, REST_RANGE


def get_file_path(directory, suffix):
    '''Generate the paths of all the given type files in the given directory.
    Args:
    - directory: The directory that contains the files you need.
    - suffix: The suffix of the file type.

    Returns:
    - path: The file path.
    - file: The file name.
    '''
    for root, _, files in os.walk(directory):
        for file_name in files:
            current_suffix = os.path.splitext(file_name)[1]

            if current_suffix != suffix:
                continue

            path = root + "/" + file_name
            yield path, file_name


def get_phrases(note_list):
    ''' Get phrases from a note list.
    Arg:
    - note_list: A list object corresponding to an instrument.

    Return:
    - phrase_list: A list object that contains all the phrases. A phrase is
    also represented as a list.
    '''

    note_list = copy.deepcopy(note_list)

    phrase_list = []
    last_idx = 0
    current_idx = 0
    length = note_list.__len__()

    for note in note_list:

        interval = note[0]
        duration = note[1]
        rest = note[2]

        if interval > INTERVAL_THRESHOLD or interval < -INTERVAL_THRESHOLD:
            # TODO not sure if I should keep what it is, rather than make it 0
            note[0] = interval = 0
            if last_idx < current_idx:
                phrase_list.append(note_list[last_idx:current_idx])
            last_idx = current_idx
        if duration > DURATION_THRESHOLD or rest > REST_THRESHOLD:
            if duration > DURATION_RANGE:
                note[1] = duration = DURATION_THRESHOLD
            if rest > REST_THRESHOLD:
                note[2] = rest = REST_THRESHOLD
            phrase_list.append(note_list[last_idx:current_idx + 1])
            last_idx = current_idx + 1

        current_idx += 1

        if current_idx == length:
            phrase_list.append(note_list[last_idx:])

    return phrase_list


def note_to_multihot(note):
    ''' Convert a `np.array` note to a `np.array` multi-hot array.
    Arg:
    - note: A note, which is an `np.array` object.

    Return:
    - multihot_note: An multi-hot array, which is an `np.array` object.
    '''

    interval = note[0]
    duration = note[1]
    rest = note[2]

    interval_onehot = np.zeros(INTERVAL_RANGE)
    duration_onehot = np.zeros(DURATION_RANGE)
    rest_onehot = np.zeros(REST_RANGE)

    # -INTERVAL_RANGE/2 is mapped to 0
    # 0 is mapped to 20
    # -20 is mapped to 0
    interval_onehot[int(interval + (INTERVAL_RANGE - 1) / 2)] = 1
    # duration == 1 is mapped to 0
    duration_onehot[int(duration - 1)] = 1
    # rest == 0 is mapped to 0
    rest_onehot[int(rest)] = 1

    multihot_note = np.concatenate((interval_onehot, duration_onehot,
                                    rest_onehot))

    return multihot_note


def phrase_to_multihot(phrase):
    ''' Convert a list of notes to a list of `np.array` multi-hot arrays.
    Arg:
    - phrase: A phrase, which is a list that contains many `np.array` objects.

    Return:
    - multihot_notes: A list that contains many multi-hot arrays.
    '''
    multihot_notes = list(map(note_to_multihot, phrase))
    return multihot_notes


def multihot_to_note(multihot_note):
    ''' Convert multi-hot array to an `np.array` note.
    Arg:
    - multihot_note: A multi-hot array that represents a note.

    Return:
    - note: An `np.array` note.
    '''

    interval_onehot = multihot_note[:INTERVAL_RANGE]
    duration_onehot = multihot_note[INTERVAL_RANGE:INTERVAL_RANGE +
                                    DURATION_RANGE]
    rest_onehot = multihot_note[INTERVAL_RANGE + DURATION_RANGE:]

    interval = np.dot(interval_onehot,
                      np.arange(INTERVAL_RANGE)) - (INTERVAL_RANGE - 1) / 2
    duration = np.dot(duration_onehot, np.arange(DURATION_RANGE)) + 1
    rest = np.dot(rest_onehot, np.arange(REST_RANGE))

    # Note that in midi_data, note_to_array_for_instrument uses np.int8 too.
    note = np.array([interval, duration, rest], dtype=np.int8)
    return note


def dump_phrase_data(phrase_data, data_num, path):
    ''' Dump the given phrase data to a pickle file.
    Args:
    - phrase_data: A `list` object that contains several phrases. A phrase is
      also a `list` object.
    - data_num: The number of phrase_data, which is used as the file name.
    - path: The output path. Note that this function will not help make the
      directory.
    '''
    data_path = path + '/' + str(data_num) + '.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump(phrase_data, f)


def construct_dataset(path, proportion):
    ''' Construct dataset by creating subfolders: training, validation, test.
    Args:
    - path: The path to the exist dataset folder.
    - proportion: The proportion of training set, validation set and test set.
    It can be a list or a tuple, which contains three numbers. The proportion
    of these three numbers is the proportion of the three sets.
    '''

    # Set random seed
    random.seed(42)

    # Obtain file list
    file_list = list(os.walk(path))[0][2]
    random.shuffle(file_list)

    # Calculate three proportions
    prop_sum = sum(proportion)
    train_prop = proportion[0] / prop_sum
    val_prop = proportion[1] / prop_sum

    # obtain the number of
    n_files = file_list.__len__()
    n_train = round(n_files * train_prop)
    n_val = round(n_files * val_prop)

    # create directory if it does not exist.
    if not os.path.exists(path + '/train'):
        os.makedirs(path + '/train')
    if not os.path.exists(path + '/valid'):
        os.makedirs(path + '/valid')
    if not os.path.exists(path + '/test'):
        os.makedirs(path + '/test')

    idx = 0
    for file_name in file_list:
        if idx < n_train:
            os.rename(path + '/' + file_name, path + '/train/' + file_name)
        elif idx >= n_train and idx < n_train + n_val:
            os.rename(path + '/' + file_name, path + '/valid/' + file_name)
        else:
            os.rename(path + '/' + file_name, path + '/test/' + file_name)
        idx += 1


def gen_batch(dataset_path, n_steps, batch_size, overlap=False, loop=True):
    ''' Generate batch data from given dataset.
    Args:
    - dataset_path: The path to the dataset.
    - n_steps: The steps of RNN.
    - batch_size: batch size.
    - overlap: Whether use overlap sampling or not.
    - loop: Whether looping the sampling process or not.

    Returns:
    - X_batch: The batch dataset as inputs.
    - y_batch: The batch dataset as targets.
    '''

    def select_duration_for_phrase(multihot_phrase):
        def select_duration_onehot(multihot_note):
            duration_onehot = multihot_note[INTERVAL_RANGE:INTERVAL_RANGE +
                                            DURATION_RANGE]
            return duration_onehot

        duration_onehoe_phrase = list(
            map(select_duration_onehot, multihot_phrase))
        return duration_onehoe_phrase

    def select_rest_for_phrase(multihot_phrase):
        def select_rest_onehot(multihot_note):
            rest_onehot = multihot_note[INTERVAL_RANGE + DURATION_RANGE:]
            return rest_onehot

        rest_onehoe_phrase = list(map(select_rest_onehot, multihot_phrase))
        return rest_onehoe_phrase

    def get_slices(n_steps, idx):
        if overlap:
            return slice(idx, idx + n_steps), slice(idx + 1, idx + n_steps + 1)
        else:
            return slice(idx * n_steps, (idx + 1) * n_steps), slice(
                idx * n_steps + 1, (idx + 1) * n_steps + 1)

    def get_loop_condition(n_steps, idx, phrase):
        if overlap:
            return idx + n_steps + 1 <= phrase.__len__()
        else:
            return (idx + 1) * n_steps + 1 <= phrase.__len__()

    while True:
        X_batch = []
        duration_batch = []
        rest_batch = []
        for data_path, data_file in get_file_path(dataset_path, '.pkl'):
            with open(data_path, 'rb') as f:
                phrase_data = pickle.load(f)
            for phrase in phrase_data:
                idx = 0
                while get_loop_condition(n_steps, idx, phrase):
                    if X_batch.__len__() == batch_size:
                        X_batch = np.array(X_batch)
                        duration_batch = np.array(duration_batch)
                        rest_batch = np.array(rest_batch)
                        yield (X_batch, {
                            'duration_output': duration_batch,
                            'rest_output': rest_batch
                        })
                        X_batch = []
                        duration_batch = []
                        rest_batch = []
                    slice_x, slice_y = get_slices(n_steps, idx)
                    X_batch.append(phrase_to_multihot(phrase[slice_x]))
                    duration_batch.append(
                        select_duration_for_phrase(
                            phrase_to_multihot(phrase[slice_y])))
                    rest_batch.append(
                        select_rest_for_phrase(
                            phrase_to_multihot(phrase[slice_y])))
                    idx += 1

        if loop is False:
            break
