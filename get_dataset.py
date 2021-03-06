# -*-coding: utf-8-*-
# Author: Yilin Zhang

import os

from xrlib.midi_data import MidiData
from xrlib.utils import get_file_path, get_phrases, dump_phrase_data, construct_dataset
from xrlib.configs import PROCESSED_MIDI_PATH, LENGTH_LIMIT, DATASET_PATH

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
