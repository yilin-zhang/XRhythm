# -*-coding: utf-8-*-
# Author: Yilin Zhang

from midi_data import MidiData
from utils import get_file_path
import pickle
import os

# The midi path that contains midi with melody extracted
PROCESSED_MIDI_PATH = './processed_midi'
# TODO LENGTH_LIMIT might be changed later
LENGTH_LIMIT = 20
PHRASE_PATH = './dataset/'


def dump_phrase_data(phrase_data, data_num, path):
    data_path = path + str(data_num) + '.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump(phrase_data, f)


# create directory if it does not exist.
if not os.path.exists(PHRASE_PATH):
    os.makedirs(PHRASE_PATH)

# ISSUE This makes it hard to generate y (target), so it should be fixed
# in this file.
# One possible solution is generating all the phrases without doing
# any pruning. In this case, I can write a function at the model file to
# generate X and y.
# Another solution is generating both X and y in this stage, but the problem
# is that the dataset takes double space since X and y contain almost the same
# notes.
# For now I think the former one is better.
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
        phrases = MidiData.get_phrases(note_list)
        for phrase in phrases:
            if phrase.__len__() < LENGTH_LIMIT + 1:
                continue
            else:
                phrase_data.append(phrase)

    if phrase_data.__len__() >= 1000:
        dump_phrase_data(phrase_data, data_num, PHRASE_PATH)
        data_num += 1
        phrase_data = []
