# -*-coding: utf-8-*-
# Author: Yilin Zhang

from keras.models import load_model
from generate import generate_midi_from_pitch_list

# Modern Music
# model_path = './models/201905310621/saved-model-50-0.98.hdf5'
# Xi'an Drum Music
model_path = './models/201905311043/saved-model-60-0.62.hdf5'

output_path = './outputs'

model = load_model(model_path)

# Define pitches
c = 60
d = 62
e = 64
f = 65
g = 67
a = 69
b = 71
c2 = 72
d2 = 74
e2 = 76
f2 = 77
g2 = 79

# Define a pitch list that contain bar information (Feng Jin Bei)
fjb = [[e, g, e, d, g, g, a], [e, e, g, d, e, g], [g, g, g, g, g, a],
       [d, e, d, g], [d2, d2, e2, d2, e2, d2], [b, a, d2, g2],
       [e2, d2, a, d2, e2], [b, b, d2, a, b, a, g]]

# Generate melody using the list above (experimental!)
# generate_midi_from_bar_pitch_list(model, fjb, output_path, n_outputs=5)

# Define pitch list (Feng Jin Bei)
fjb_pitch_list = [
    e, g, e, d, g, g, a, e, e, g, d, e, g, g, g, g, g, g, a, d, e, d, g, d2,
    d2, e2, d2, e2, d2, b, a, d2, g2, e2, d2, a, d2, e2, b, b, d2, a, b, a, g
]

# Generate melody using the list above
generate_midi_from_pitch_list(
    model, fjb_pitch_list, output_path, n_outputs=10, randomness=False)
