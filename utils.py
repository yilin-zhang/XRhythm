# -*-coding: utf-8-*-
# Author: Yilin Zhang

import os

RAW_MIDI_PATH = './raw_midi'
PROCESSED_MIDI_PATH = './processed_midi'
DATASET_PATH = './dataset'
LENGTH_LIMIT = 20

# Note that these three constants are also written in midi_data.py
# remember to change them when change these.
INTERVAL_RANGE = 20 * 2
DURATION_RANGE = 40
REST_RANGE = 32


def get_file_path(directory, suffix):
    '''Generate the paths of all the given type files in the given directory.
    Arg:
    - directory: The directory that contains the files you need.
    - suffix: The suffix of the file type.

    Return:
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
