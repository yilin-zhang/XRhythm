# -*-coding: utf-8-*-
# Author: Yilin Zhang

import os


def get_midi_path(directory):
    '''Generate the paths of all the midi files in the given directory.
    Arg:
    - directory: The directory that contains midi files.

    Return:
    - midi_path: The midi file path.
    - midi_file: The midi file name.
    '''
    for root, _, files in os.walk(directory):
        for midi_file in files:
            suffix = os.path.splitext(midi_file)[1]

            if suffix != '.mid':
                continue

            midi_path = root + "/" + midi_file
            yield midi_path, midi_file
