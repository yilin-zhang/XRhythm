# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
from keras.models import load_model
import numpy as np

# Internal imports
from midi_data import MidiData
from utils import multihot_to_note
from utils import LENGTH_LIMIT, INTERVAL_RANGE, DURATION_RANGE, REST_RANGE

MODEL_PATH = './models/20190426/lstm_model.h5'


def predicted_note_to_multihot(predicted_note):
    ''' Convert the float type predicted note to multi-hot array format.

    Arg:
    - predicted_note: The predicted note array, which is float type.

    Return:
    - multihot_note: A multi-hot array.
    '''
    interval_part = predicted_note[:INTERVAL_RANGE]
    duration_part = predicted_note[INTERVAL_RANGE:INTERVAL_RANGE +
                                   DURATION_RANGE]
    rest_part = predicted_note[INTERVAL_RANGE + DURATION_RANGE:]

    interval_onehot = np.zeros(INTERVAL_RANGE)
    duration_onehot = np.zeros(DURATION_RANGE)
    rest_onehot = np.zeros(REST_RANGE)

    interval_onehot[np.argmax(interval_part)] = 1
    duration_onehot[np.argmax(duration_part)] = 1
    rest_onehot[np.argmax(rest_part)] = 1

    multihot_note = np.concatenate((interval_onehot, duration_onehot,
                                    rest_onehot))

    return multihot_note


def construct_steps(multihot_notes):
    ''' Construct all the steps for input, which may contain random arrays

    Arg:
    - multihot_notes: A list that contains multihot_notes, whose length
    can be less than the number of steps.

    Return:
    - model_input: The input array that contains both random arrays and
    actual multi-hot notes.
    '''

    def generate_random_array():
        ''' Generate random array for model's input.'''
        return np.random.rand(INTERVAL_RANGE + DURATION_RANGE + REST_RANGE)

    n_random_steps = LENGTH_LIMIT - multihot_notes.__len__()
    random_steps = []

    # Only when the multihot_notes less than the number of steps,
    # generate random steps
    if n_random_steps > 0:
        for _ in range(n_random_steps):
            random_steps.append(generate_random_array())

    model_input = np.array(random_steps + multihot_notes)

    return model_input


def generate_note_list_from_intervals(model, interval_list):
    ''' Generate note_list (melody) form an interval list.

    Args:
    - model: LSTM model.
    - interval_list: A list of intervals.

    Return:
    - note_list: A list of notes, which represent melody.
    '''

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

    def construct_new_input(last_predicted_note, current_interval_onehot):
        ''' Construct the float type predicted note for a new input for prediction
        purpose.

        Args:
        - last_predicted_note: The predicted note array from the last step.
        - currrent_interval_onehot: The current interval, which is a one-hot array.

        Return:
        - multihot_note: A multi-hot array.
        '''

        last_multihot_note = predicted_note_to_multihot(last_predicted_note)
        duration_onehot = last_multihot_note[INTERVAL_RANGE:INTERVAL_RANGE +
                                             DURATION_RANGE]
        rest_onehot = last_multihot_note[INTERVAL_RANGE + DURATION_RANGE:]

        multihot_note = np.concatenate((current_interval_onehot,
                                        duration_onehot, rest_onehot))

        return multihot_note

    model_input = construct_steps([])
    for step in range(interval_list.__len__()):
        prediction = model.predict(
            model_input.reshape(1, LENGTH_LIMIT,
                                INTERVAL_RANGE + DURATION_RANGE + REST_RANGE),
            batch_size=1,
            verbose=0)
        # the first index is 0 because the batch size is 1.
        # the second index is -1 because we only need the last output.
        new_input = construct_new_input(
            prediction[0][-1], interval_to_onehot(interval_list[step]))
        # print(new_input)
        model_input = np.vstack([model_input[1:], new_input])

    # After the for loop above, the model_input is exactly the multi-hot notes
    # we want.
    multihot_notes = model_input[-interval_list.__len__():]

    # Generate note list.
    note_list = []
    for multihot_note in multihot_notes:
        note_list.append(multihot_to_note(multihot_note))

    return note_list


if __name__ == '__main__':
    model = load_model(MODEL_PATH)

    interval_list = [3, 6, -2, 4, 0]
    note_list = generate_note_list_from_intervals(model, interval_list)
    print(note_list)
    # melody = MidiData.note_list_to_mididata(note_list)
    # melody.write('./melody.mid')

    # The following part is for test
    # note = np.array([-2, 5, 3])
    # phrase = [note] * 20
    # multihot_input = np.array(phrase_to_multihot(phrase)).reshape(
    #     1, LENGTH_LIMIT, INTERVAL_RANGE + DURATION_RANGE + REST_RANGE)
    # predictions = model.predict(multihot_input, batch_size=1, verbose=0)
    # print(predictions)
