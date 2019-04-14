# -*-coding: utf-8-*-
# Author: Yilin Zhang

from midi_data import MidiData
from utils import get_midi_path
import pickle

# The midi path that contains midi with melody extracted
PROCESSED_MIDI_PATH = './processed_midi'

phrase_lengths = []
for midi_path, midi_file in get_midi_path(PROCESSED_MIDI_PATH):
    try:
        midi = MidiData(midi_path, res=1 / 16)
    except:
        print('corrupted: ' + midi_path)
        continue

    print('processing: ' + midi_path)
    instrument_list = midi.note_to_array()

    for note_list in instrument_list:
        # phrase_lists.append(MidiData.get_phrases(note_list))
        phrases = MidiData.get_phrases(note_list)
        for phrase in phrases:
            phrase_lengths.append(phrase.__len__())

with open('./pickle/phrase_lengths.pkl', 'wb') as f:
    pickle.dump(phrase_lengths, f)
