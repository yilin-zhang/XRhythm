# -*-coding: utf-8-*-
# Author: Yilin Zhang

import os
from xrlib.midi_data import MidiData
from xrlib.utils import get_file_path

# import constants
from xrlib.configs import XADRUM_MIDI_PATH, PROCESSED_XADRUM_MIDI_PATH
from xrlib.configs import RESOLUTION, FILTER_RESOLUTION

for midi_path, midi_file in get_file_path(XADRUM_MIDI_PATH, '.mid'):
    try:
        midi = MidiData(midi_path, res=RESOLUTION)
    except:
        print('corrupted: ' + midi_path)
        continue

    # Process midi
    midi.quantize(filter_res=FILTER_RESOLUTION)

    # Construct output path
    path_components = midi_path.split('/')[XADRUM_MIDI_PATH.split('/').
                                           __len__():]
    path_components = PROCESSED_XADRUM_MIDI_PATH.split('/') + path_components
    output_path = '/'.join(path_components)

    # Output midi file
    try:
        midi.write(output_path)
    except FileNotFoundError:
        os.makedirs(os.path.split(output_path)[0])
        midi.write(output_path)
