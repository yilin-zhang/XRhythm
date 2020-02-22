# XRhythm

This project is a short term result of a long term research project. The Xi'an
Drum Music MIDI dataset is not public for now.

## Usage

Prerequisite
- Python >= 3.6
- `pretty_midi`
- `keras`

Please check `example.py` to know how to use it.


### Data Pre-processing

Create a `midi` folder in the project directory. Create `raw_midi` and
`xadrum_midi` folders in `midi` folder.

```
midi
  ├─── raw_midi
  └─── xadrum_midi
```

Put MIDI files into the 2 folders. Put modern music dataset (Lakh MIDI Dataset)
into `raw_midi`, Xi'an Drum Music dataset into `xadrum_midi`.

Run `preprocess_midi.py` and `preprocess_xadrum.py`.

After the pre-processing, `processed_midi` and `xadrum_processed_midi` will be
created.

```
midi
  ├─── raw_midi
  ├─── processed_midi
  ├─── xadrum_midi
  └─── xadrum_processed_midi
```
### Data Serialization

Run `get_dataset.py` and `get_xadrum_dataset.py` to serialize the data. The
`datasets` folder will be created.

```
datasets
  ├─── dataset
  └─── xadrum_dataset
```

### Model Training
Run `train_modern.py` first to get a trained model, and `models` folder will be
created.

Then run `train_xadrum.py` based on the previous model (you should change the
path inside of `train_xadrum.py`).
