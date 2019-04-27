# -*-coding: utf-8-*-
# Author: Yilin Zhang

import os
from midi_data import MidiData
from utils import get_file_path

# import constants
from utils import RAW_MIDI_PATH, PROCESSED_MIDI_PATH
from utils import RESOLUTION, FILTER_RESOLUTION

for midi_path, midi_file in get_file_path(RAW_MIDI_PATH, '.mid'):
    try:
        midi = MidiData(midi_path, res=RESOLUTION)
    except:
        print('corrupted: ' + midi_path)
        continue

    # Process midi
    midi.drop_drum()
    midi.quantize(filter_res=FILTER_RESOLUTION)
    midi.skyline()

    # Construct output path
    path_components = midi_path.split('/')[RAW_MIDI_PATH.split('/').__len__():]
    path_components = PROCESSED_MIDI_PATH.split('/') + path_components
    output_path = '/'.join(path_components)

    # Output midi file
    try:
        midi.write(output_path)
    except FileNotFoundError:
        os.makedirs(os.path.split(output_path)[0])
        midi.write(output_path)
