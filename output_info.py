# -*-coding: utf-8-*-
# Author: Yilin Zhang

from midi_data import MidiData
from utils import get_midi_path
import pickle

PROCESSED_MIDI_PATH = './processed_midi'

itv_freq = {}
dur_freq = {}
res_freq = {}

for midi_path, midi_file in get_midi_path(PROCESSED_MIDI_PATH):
    try:
        midi = MidiData(midi_path, res=1 / 16)
    except:
        print('corrupted: ' + midi_path + '\n')
        continue

    instrument_list = midi.note_to_array()

    # Update and print:
    # the biggest interval
    # the longest duration
    # the longest rest
    big_pos_itv = 0
    big_neg_itv = 0
    long_dur = 0
    long_res = 0
    for note_list in instrument_list:
        for note in note_list:
            # frequency
            if note[0] in itv_freq:
                itv_freq[note[0]] += 1
            else:
                itv_freq[note[0]] = 1

            if note[1] in dur_freq:
                dur_freq[note[1]] += 1
            else:
                dur_freq[note[1]] = 1

            if note[2] in res_freq:
                res_freq[note[2]] += 1
            else:
                res_freq[note[2]] = 1

            # largest values
            if note[0] > big_pos_itv:
                big_pos_itv = note[0]
            elif note[0] < big_neg_itv:
                big_neg_itv = note[0]

            if note[1] > long_dur:
                long_dur = note[1]

            if note[2] > long_res:
                long_res = note[2]

    print("midi_path: ", midi_path)
    print("big_pos_itv:", big_pos_itv)
    print("big_neg_itv:", big_neg_itv)
    print("long_dur:", long_dur)
    print("long_res:", long_res, "\n")

with open('./pickle/itv_freq.pkl', 'wb') as f:
    pickle.dump(itv_freq, f)

with open('./pickle/dur_freq.pkl', 'wb') as f:
    pickle.dump(dur_freq, f)

with open('./pickle/res_freq.pkl', 'wb') as f:
    pickle.dump(res_freq, f)
