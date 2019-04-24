# -*-coding: utf-8-*-
# Author: Yilin Zhang

from midi_data import MidiData
from utils import get_file_path, get_phrases
import pickle
import os
import random

from utils import PROCESSED_MIDI_PATH, LENGTH_LIMIT, DATASET_PATH


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


if __name__ == '__main__':
    # create directory if it does not exist.
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    # process all the midi files.
    data_num = 0
    phrase_data = []
    for midi_path, midi_file in get_file_path(PROCESSED_MIDI_PATH, '.mid'):
        try:
            midi = MidiData(midi_path, res=1 / 16)
        except:
            print('corrupted: ' + midi_path)
            continue

        instrument_list = midi.note_to_array()

        for note_list in instrument_list:
            phrases = get_phrases(note_list)
            for phrase in phrases:
                if phrase.__len__() < LENGTH_LIMIT + 1:
                    continue
                else:
                    phrase_data.append(phrase)

        if phrase_data.__len__() >= 1000:
            dump_phrase_data(phrase_data, data_num, DATASET_PATH)
            data_num += 1
            phrase_data = []

    construct_dataset(DATASET_PATH, (0.7, 0.15, 0.15))
