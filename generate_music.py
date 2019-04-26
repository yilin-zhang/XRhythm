# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
from keras.models import load_model
import numpy as np

# Internal imports
from midi_data import MidiData
from utils import multihot_to_note
from utils import LENGTH_LIMIT, INTERVAL_RANGE, DURATION_RANGE, REST_RANGE

# MODEL_PATH = './models/20190426/lstm_model.h5'
MODEL_PATH = './models/20190426-dev/lstm_model.h5'


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


def generate_note_list_from_interval_list(model, interval_list):
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

    multihot_notes = []
    model_input = init_steps()
    for step in range(interval_list.__len__()):
        prediction = model.predict(
            model_input.reshape(1, LENGTH_LIMIT,
                                INTERVAL_RANGE + DURATION_RANGE + REST_RANGE),
            batch_size=1,
            verbose=0)
        # the first index is 0 because the batch size is 1.
        # the second index is -1 because we only need the last output.
        last_predicted_duration = prediction[0][0][-1]
        last_predicted_rest = prediction[1][0][-1]
        new_input = construct_new_input(
            last_predicted_duration, last_predicted_rest,
            interval_to_onehot(interval_list[step]))
        # Add new_input to multihot_notes
        multihot_notes.append(new_input)
        # Reconstruct model_input
        model_input = np.vstack([model_input[1:], new_input])

    # Generate note list.
    note_list = []
    for multihot_note in multihot_notes:
        note_list.append(multihot_to_note(multihot_note))

    return note_list


if __name__ == '__main__':
    model = load_model(MODEL_PATH)

    interval_list = [3, 6, -2, 4, 0] * 5
    note_list = generate_note_list_from_interval_list(model, interval_list)
    print(note_list)
    # melody = MidiData.note_list_to_mididata(note_list)
    # melody.write('./melody.mid')

    # The following part is for test
    # note = np.array([-2, 5, 3])
    # phrase = [note] * 20
    # multihot_input = np.array(phrase_to_multihot(phrase)).reshape(
    # 1, LENGTH_LIMIT, INTERVAL_RANGE + DURATION_RANGE + REST_RANGE)
    # predictions = model.predict(multihot_input, batch_size=1, verbose=0)
    # print(predictions)
