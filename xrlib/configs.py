# -*-coding: utf-8-*-
# Author: Yilin Zhang

RAW_MIDI_PATH = './midi/raw_midi'
XADRUM_MIDI_PATH = './midi/xadrum_midi'
PROCESSED_MIDI_PATH = './midi/processed_midi'
PROCESSED_XADRUM_MIDI_PATH = './midi/xadrum_processed_midi'
DATASET_PATH = './datasets/dataset'  # path to modern music dataset
XADRUM_DATASET_PATH = './datasets/xadrum_dataset'
MODEL_PATH = './models'
LOG_PATH = './logs'

RESOLUTION = 1 / 16
FILTER_RESOLUTION = 1 / 16

LENGTH_LIMIT = 16

INTERVAL_THRESHOLD = 12
DURATION_THRESHOLD = 16
REST_THRESHOLD = 16

INTERVAL_RANGE = INTERVAL_THRESHOLD * 2 + 1
DURATION_RANGE = DURATION_THRESHOLD  # duration can not be 0
REST_RANGE = REST_THRESHOLD + 1  # rest can be 0
