# -*-coding: utf-8-*-
# Author: Yilin Zhang

import os
import copy
import numpy as np

RAW_MIDI_PATH = './raw_midi'
PROCESSED_MIDI_PATH = './processed_midi'
DATASET_PATH = './dataset'
LENGTH_LIMIT = 20

INTERVAL_THRESHOLD = 20
DURATION_THRESHOLD = 40
REST_THRESHOLD = 32

INTERVAL_RANGE = INTERVAL_THRESHOLD * 2 + 1
DURATION_RANGE = DURATION_THRESHOLD  # duration can not be 0
REST_RANGE = REST_THRESHOLD + 1  # rest can be 0


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

        if current_idx == 0:
            current_idx += 1
            continue

        interval = note[0]
        duration = note[1]
        rest = note[2]

        if interval > INTERVAL_THRESHOLD or interval < -INTERVAL_THRESHOLD:
            # TODO not sure if I should keep what it is, rather than make it 0
            interval = 0
            note[0] = interval
            phrase_list.append(note_list[last_idx:current_idx])
            last_idx = current_idx
            # print('phrase break: interval')
        elif duration > DURATION_THRESHOLD:
            duration = DURATION_THRESHOLD
            note[1] = duration
            phrase_list.append(note_list[last_idx:current_idx + 1])
            last_idx = current_idx + 1
            # print('phrase break: duration')
        elif rest > REST_THRESHOLD:
            rest = REST_THRESHOLD
            note[2] = rest
            phrase_list.append(note_list[last_idx:current_idx + 1])
            last_idx = current_idx + 1
            # print('phrase break: rest')

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
    - note: A note, which is an `np.array` object.

    Return:
    - multihot_note: An multi-hot array, which is an `np.array` object.
    '''

    multihot_phrase = copy.deepcopy(phrase)
    idx = 0
    for note in phrase:
        multihot_note = note_to_multihot(note)
        multihot_phrase[idx] = multihot_note
        idx += 1
    return multihot_phrase


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

    interval = np.dot(interval_onehot, np.arange(INTERVAL_RANGE))
    duration = np.dot(duration_onehot, np.arange(DURATION_RANGE)) + 1
    rest = np.dot(rest_onehot,
                  np.arange(REST_RANGE)) - (INTERVAL_RANGE - 1) / 2

    # Note that in midi_data, note_to_array_for_instrument uses np.int8 too.
    note = np.zeros(3, dtype=np.int8)
    note[0] = interval
    note[1] = duration
    note[2] = rest

    return note
