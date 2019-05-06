# -*-coding: utf-8-*-
# Author: Yilin Zhang

RAW_MIDI_PATH = './raw_midi'
PROCESSED_MIDI_PATH = './processed_midi'
XADRUM_MIDI_PATH = './xadrum_midi'
DATASET_PATH = './dataset'  # path to modern music dataset
XADRUM_DATASET_PATH = './xadrum_dataset'

RESOLUTION = 1 / 16
FILTER_RESOLUTION = 1 / 16

LENGTH_LIMIT = 16

INTERVAL_THRESHOLD = 12
DURATION_THRESHOLD = 16
REST_THRESHOLD = 16

INTERVAL_RANGE = INTERVAL_THRESHOLD * 2 + 1
DURATION_RANGE = DURATION_THRESHOLD  # duration can not be 0
REST_RANGE = REST_THRESHOLD + 1  # rest can be 0
