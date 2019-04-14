# -*-coding: utf-8-*-
# Author: Yilin Zhang

from midi_data import MidiData
from utils import get_midi_path
import pickle

# The midi path that contains midi with melody extracted
PROCESSED_MIDI_PATH = './processed_midi'
LENGTH_LIMIT = 10

phrase_data = []
for midi_path, midi_file in get_midi_path(PROCESSED_MIDI_PATH):
    try:
        midi = MidiData(midi_path, res=1 / 16)
    except:
        print('corrupted: ' + midi_path)
        continue

    print('processing: ' + midi_path)
    instrument_list = midi.note_to_array()

    for note_list in instrument_list:
        phrases = MidiData.get_phrases(note_list)
        for phrase in phrases:
            phrase_len = phrase.__len__()
            # TODO phrase_len should be a fixed number
            # abandon phrase when phrase_len < LENGTH_LIMIT
            # cut phrase into pieces when phrase_len > LENGTH_LIMIT
            if phrase_len >= 4 and phrase_len <= 64:
                phrase_data.append(phrase)

with open('./pickle/phrase_data.pkl', 'wb') as f:
    pickle.dump(phrase_data, f)
