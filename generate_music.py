# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
from keras.models import load_model
import numpy as np
import copy
import os
from datetime import datetime

# Internal imports
from midi_data import MidiData
from utils import multihot_to_note
from utils import LENGTH_LIMIT, INTERVAL_RANGE, DURATION_RANGE, REST_RANGE
from utils import RESOLUTION


def generate_note_lists_from_interval_list(model, interval_list, n_outputs):
    ''' Generate note_list (melody) form an interval list.

    Args:
    - model: LSTM model.
    - interval_list: A list of intervals.
    - n_outputs: The output number.

    Return:
    - note_lists: A list of note_list, the first dimension is n_outputs.
    '''

    def predicted_note_to_multihot(predicted_duration, predicted_rest):
        ''' Convert the float type predicted note to multi-hot array format.

        Args:
        - predicted_duration: The predicted duration array, whose element is float
        type.
        - predicted_rest: The predicted rest array.

        Return:
        - multihot_partial_note: A multi-hot array.
        '''

        duration_onehot = np.zeros(DURATION_RANGE)
        rest_onehot = np.zeros(REST_RANGE)

        duration_onehot[np.argmax(predicted_duration)] = 1
        rest_onehot[np.argmax(predicted_rest)] = 1

        multihot_partial_note = np.concatenate((duration_onehot, rest_onehot))

        return multihot_partial_note

    def init_steps():
        ''' Randomly initialize all steps for model's input.

        Return:
        - model_input: The input array that contains random arrays.
        '''

        def generate_random_array():
            ''' Generate random array for model's input.'''
            return np.random.rand(INTERVAL_RANGE + DURATION_RANGE + REST_RANGE)

        random_steps = []

        # Only when the multihot_notes less than the number of steps,
        # generate random steps
        for _ in range(LENGTH_LIMIT):
            random_steps.append(generate_random_array())

        model_input = np.array(random_steps)

        return model_input

    def interval_to_onehot(interval):
        ''' Convert interval to a one-hot array.

        Arg:
        - interval: An integer number.

        Return:
        - interval_onehot: A one-hot array.
        '''

        # The next two lines are the same as the fuction note_to_multihot
        interval_onehot = np.zeros(INTERVAL_RANGE)
        interval_onehot[int(interval + (INTERVAL_RANGE - 1) / 2)] = 1

        return interval_onehot

    def construct_new_input(last_predicted_duration, last_predicted_rest,
                            current_interval_onehot):
        ''' Construct the float type predicted attributes for a new input for
        prediction purpose.

        Args:
        - last_predicted_duration: The last predicted duration array.
        - last_predicted_rest: The last predicted rest array.

        Return:
        - multihot_note: A multi-hot array.
        '''

        # Convert the raw output array to multi-hot array
        last_multihot_partial_note = predicted_note_to_multihot(
            last_predicted_duration, last_predicted_rest)

        # Concatenate those three one-hot arrays
        multihot_note = np.concatenate((current_interval_onehot,
                                        last_multihot_partial_note))

        return multihot_note

    batch_size = n_outputs
    batch_multihot_notes = [[] for _ in range(batch_size)]
    model_input = np.array([init_steps() for _ in range(batch_size)])
    for step in range(interval_list.__len__()):
        prediction = model.predict(
            model_input.reshape(batch_size, LENGTH_LIMIT,
                                INTERVAL_RANGE + DURATION_RANGE + REST_RANGE),
            batch_size=batch_size,
            verbose=0)
        # the second index is 0 because the batch size is 1.
        # the third index is -1 because we only need the last output.
        for i in range(batch_size):
            last_predicted_duration = prediction[0][i][-1]
            last_predicted_rest = prediction[1][i][-1]
            new_input = construct_new_input(
                last_predicted_duration, last_predicted_rest,
                interval_to_onehot(interval_list[step]))
            # Add new_input to multihot_notes
            batch_multihot_notes[i].append(new_input)
            # Reconstruct model_input
            model_input[i] = np.vstack([model_input[i][1:], new_input])

    # Generate note list.
    note_lists = [[] for _ in range(batch_size)]
    for idx, multihot_notes in enumerate(batch_multihot_notes):
        for multihot_note in multihot_notes:
            note_lists[idx].append(multihot_to_note(multihot_note))

    return note_lists


def generate_note_lists_from_bar_interval_list(model, bar_interval_list,
                                               n_outputs):
    ''' Generate music based on given bars information.
    Args:
    - model: LSTM model
    - bar_interval_list: A list whose sublists are interval lists.
    - n_outputs: The number of generated note lists.

    Return:
    - note_list: A list of notes, which represents melody.
    '''

    # 1st Step: generate the whole melody
    # 2nd Step: scale durations and rests
    #           if duration is less then the minimum value,
    #           set it as the minimum value.

    def scale_note_list(note_list):
        ''' Scale the note list to fit one bar. '''

        def cal_total_units(note_list):
            total_units = 0
            for note in note_list:
                duration = note[1]
                rest = note[2]
                total_units += duration + rest
            return total_units

        note_list = copy.deepcopy(note_list)
        units_per_bar = 1 / RESOLUTION

        # The number of notes can note be more than
        # units_per_bar, otherwise some note can not sound.
        if note_list.__len__() > units_per_bar:
            raise Exception('Too many notes!')

        # Calculate how many units all the notes have.
        total_units = cal_total_units(note_list)
        ratio = units_per_bar / total_units

        # Apply the ratio to all the notes.
        # If duration == 0, set it to ensure the note can sound.
        for note in note_list:
            duration = round(note[1] * ratio)
            rest = round(note[2] * ratio)
            if duration == 0:
                duration = 1
            note[1] = duration
            note[2] = rest

        # Calculate again
        total_units = cal_total_units(note_list)

        # Obtain the differnce.
        diff_units = total_units - units_per_bar

        # If difference exits, eliminate it.
        # The code below is a mess...
        # TODO Improve it!
        if diff_units != 0:
            if diff_units > 0:
                # Add one unit per loop
                increment = -1
            else:
                # Substract one unit per loop
                increment = 1

            i = 0
            rest_idx_list = []
            duration_idx_list = []
            work_on_rest = True

            while i < abs(diff_units):
                # work on rest first
                if work_on_rest:
                    max_rest_idx = 0
                    max_rest = 0
                    # Find the maximum rest value
                    # and the corresponding index.
                    for idx, note in enumerate(note_list):
                        if idx in rest_idx_list:
                            continue
                        rest = note[2]
                        if rest > max_rest:
                            max_rest = rest
                            max_rest_idx = idx
                    # max_rest == 0 means now we should work on duration.
                    if max_rest == 0:
                        work_on_rest = False
                    else:
                        rest_idx_list.append(max_rest_idx)
                        i += 1
                # work on duration
                else:
                    max_duration_idx = 0
                    max_duration = 0
                    # Find the maximum duration value
                    # and the corresponding index.
                    for idx, note in enumerate(note_list):
                        if idx in duration_idx_list:
                            continue
                        duration = note[1]
                        if duration > max_duration:
                            max_duration = duration
                            max_duration_idx = idx
                    duration_idx_list.append(max_duration_idx)
                    i += 1

            # Do increment for all the notes whose indexes are in
            # those two lists
            for idx in rest_idx_list:
                note_list[idx][2] += increment
            for idx in duration_idx_list:
                note_list[idx][1] += increment

        return note_list

    # Flatten bar_interval_list
    interval_list = [itv for bar in bar_interval_list for itv in bar]

    raw_note_lists = generate_note_lists_from_interval_list(
        model, interval_list, n_outputs)

    # Scale notes' durations and rests based on bars.
    note_lists = [[] for _ in range(n_outputs)]
    for bar in bar_interval_list:
        bar_note_lists = [[] for _ in range(n_outputs)]
        # Fetch out all the notes in the bar.
        for _ in bar:
            # pop the first note from raw_note_list and
            # push in into bar_note_list
            for idx in range(n_outputs):
                bar_note_lists[idx].append(raw_note_lists[idx].pop(0))
        # Scale these notes, and concatenate them with note_lists
        for idx in range(n_outputs):
            note_lists[idx] = note_lists[idx] + scale_note_list(
                bar_note_lists[idx])

    return note_lists


def pitch_list_to_interval_list(pitch_list):
    ''' Convert pitches (a list) to intervals (a list).
    Arg:
    - pitch_list: A list object that contains all the absolute pitces.
    TODO for now it only support integers.

    Return:
    - interval_list: A list object that contains all the intervals.
    '''

    def character_to_number(pitch_char):
        ''' Convert pitch character to number.'''
        # TODO
        pass

    last_pitch = pitch_list[0]
    interval_list = []

    for pitch in pitch_list:
        interval_list.append(pitch - last_pitch)
        last_pitch = pitch

    return interval_list


def generate_midi_from_bar_pitch_list(model,
                                      bar_pitch_list,
                                      path,
                                      n_outputs=10,
                                      tempo=100):
    ''' Generater midi file from given bar pitch list (a list whose sublist is
    pitch list.)

    Args:
    - model: LSTM model.
    - bar_pitch_list: A list whose sublist is pitch list.
    - path: MIDI file output path (should be a directory).
    - n_outputs: The number of output midi files.
    - tempo: The speed of output (beats (quarter note) per minute).
    '''
    start_pitch = bar_pitch_list[0][0]

    def bar_pitch_list_to_bar_interval_list(bar_pitch_list):
        bar_interval_list = list(
            map(pitch_list_to_interval_list, bar_pitch_list))
        # We have to modify the first interval in a bar.
        for idx, bar in enumerate(bar_interval_list):
            # Skip the first interval
            if idx == 0:
                continue
            # The current interval should be the current pitch minus the
            # previous pitch.
            bar[0] = bar_pitch_list[idx][0] - bar_pitch_list[idx - 1][-1]

        return bar_interval_list

    bar_interval_list = bar_pitch_list_to_bar_interval_list(bar_pitch_list)
    note_lists = generate_note_lists_from_bar_interval_list(
        model, bar_interval_list, n_outputs)

    time_stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    if not os.path.exists(path):
        os.makedirs(path)

    for idx, note_list in enumerate(note_lists):
        melody = MidiData.note_list_to_mididata(
            note_lists[idx],
            start_pitch=start_pitch,
            tempo=tempo,
            res=RESOLUTION)
        midi_path = path + '/' + time_stamp + '_' + str(idx) + '.mid'
        melody.write(midi_path)


if __name__ == '__main__':

    MODEL_PATH = './models/20190427-dev/lstm_model.h5'
    model = load_model(MODEL_PATH)

    pitch_list = [60, 62, 64, 65, 67]
    bar_pitch_list = [pitch_list, [60, 62, 64]]
    generate_midi_from_bar_pitch_list(
        model, bar_pitch_list, './outputs', n_outputs=3)
    # note_lists = generate_note_lists_from_interval_list(
    # model, [2, 2, 2, 1, 2], 3)

    # note_lists = generate_note_lists_from_bar_interval_list(
    # model, [[2, 2, 2], [1, 2]], 2)
    # print(note_lists)
