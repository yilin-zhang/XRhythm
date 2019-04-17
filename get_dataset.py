# -*-coding: utf-8-*-
# Author: Yilin Zhang

from midi_data import MidiData
from utils import get_midi_path
import pickle

# The midi path that contains midi with melody extracted
PROCESSED_MIDI_PATH = './processed_midi'
LENGTH_LIMIT = 20
PHRASE_PATH = './dataset/'


def dump_phrase_data(phrase_data, data_num, path):
    data_path = path + str(data_num) + '.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump(phrase_data, f)


data_num = 0
phrase_data = []
for midi_path, midi_file in get_midi_path(PROCESSED_MIDI_PATH):
    try:
        midi = MidiData(midi_path, res=1 / 16)
    except:
        print('corrupted: ' + midi_path)
        continue

    instrument_list = midi.note_to_array()

    for note_list in instrument_list:
        phrases = MidiData.get_phrases(note_list)
        for phrase in phrases:
            phrase_len = phrase.__len__()
            # TODO phrase_len should be a fixed number
            # abandon phrase when phrase_len < LENGTH_LIMIT
            # cut phrase into pieces when phrase_len > LENGTH_LIMIT
            while phrase_len >= LENGTH_LIMIT:
                phrase_data.append(phrase[:LENGTH_LIMIT])
                phrase = phrase[LENGTH_LIMIT:]
                phrase_len = phrase.__len__()

    if phrase_data.__len__() >= 1000:
        dump_phrase_data(phrase_data, data_num, PHRASE_PATH)
        data_num += 1
        phrase_data = []
