# -*-coding: utf-8-*-
# Author: Yilin Zhang

import os
from midi_data import MidiData
from utils import get_midi_path

RAW_MIDI_PATH = './raw_midi'
PROCESSED_MIDI_PATH_NAME = 'processed_midi'

for midi_path, midi_file in get_midi_path(RAW_MIDI_PATH):
    try:
        midi = MidiData(midi_path, res=1 / 16)
    except:
        print('corrupted: ' + midi_path)
        continue

    # Process midi
    midi.drop_drum()
    midi.quantize(filter_res=1 / 16)
    midi.skyline()

    # Construct output path
    path_components = midi_path.split('/')
    path_components[1] = PROCESSED_MIDI_PATH_NAME
    output_path = '/'.join(path_components)

    # Output midi file
    try:
        midi.write(output_path)
    except FileNotFoundError:
        os.makedirs(os.path.split(output_path)[0])
        midi.write(output_path)
