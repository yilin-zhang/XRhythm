# -*-coding: utf-8-*-
# Author: Yilin Zhang

from midi_data import MidiData
from utils import get_file_path, get_phrases
import pickle
import os

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
