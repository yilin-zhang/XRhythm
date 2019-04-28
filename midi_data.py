# -*-coding: utf-8-*-
# Author: Yilin Zhang

import pretty_midi
import copy
import numpy as np


class MidiData():
    ''' Contains midi data (`pretty_midi.PrettyMIDI`) and other utility
    functions.

    Internal variables:
    - self.pm: `pretty_midi.PrettyMIDI` object.
    - self.res: Resolution for midi, the unit is bar (assume 4 beats a bar).
    - self.beats: The time stamps of All the
    '''

    def __init__(self, midi, res=1 / 16):
        '''
        Args:
        - midi_path: The path to a midi file.
        - res: The resolution of the midi file, which will decide the
        resolution of quantization and note-to-array convertion. The unit is
        bar, which means res=1/4 is one beat.
        (assume the time signature is 4/4).
        '''
        if type(midi) is str:
            self.pm = pretty_midi.PrettyMIDI(midi)
        elif type(midi) is pretty_midi.PrettyMIDI:
            self.pm = midi
        self.res = res
        self.beats = self.pm.get_beats()

    def _process_per_instrument(self, method, *args):
        '''Apply all the instruments to one method.'''
        instruments = []
        for instrument in self.get_instruments():
            instruments.append(method(instrument, *args))
        self.pm.instruments = instruments
        return self

    def get_instruments(self):
        '''Obtain instruments'''
        return self.pm.instruments

    def write(self, path):
        self.pm.write(path)

    def drop_drum(self):
        ''' Drop drum instruments in the PrettyMIDI object. The modification
        is inplace.

        Return:
        - self: The `MidiData` object whose drum tracks has been dropped.
        '''

        pm_without_drum = self.pm

        instruments = [
            instrument for instrument in pm_without_drum.instruments
            if not instrument.is_drum
        ]
        pm_without_drum.instruments = instruments

        return pm_without_drum

    def quantize_for_instrument(self, instrument, filter_res):
        ''' Quantize an instrument.

        Args:
        - instrument: A `pretty_midi.Instrument` object.
        - filter_res: The resolution which indicates the minimun length of
        notes.

        Return:
        - quantized_instrument: A quantized `pretty_midi.Instrument` object.
        '''

        beats = self.beats
        res = self.res

        # convert the unit of resolution from bar to beat.
        res_coef = res * 4
        filter_coef = filter_res * 4

        # WORKAROUND There should be a more general way to do this,
        # but now I just assume the tempo of a whole song is fixed.
        sec_per_beat = float(beats[1] - beats[0])
        sec_unit = sec_per_beat * res_coef  # the time unit for quantization
        sec_filter = sec_per_beat * filter_coef  # the minimun time for a note

        quantized_instrument = copy.deepcopy(instrument)
        notes = quantized_instrument.notes

        # make sure all the notes are sorted by start time
        notes.sort(key=lambda note: note.start)

        quantized_notes = []

        for note in notes:
            start_point = note.start
            end_point = note.end

            # units stands for quantization grids
            start_unit = start_point / sec_unit
            end_unit = end_point / sec_unit

            # find the nearest grid
            quantized_start_unit = round(start_unit)
            quantized_end_unit = round(end_unit)

            # convert it back to seconds
            start_point = quantized_start_unit * sec_unit
            end_point = quantized_end_unit * sec_unit

            # filter out short notes
            if end_point - start_point < sec_filter:
                continue
            else:
                quantized_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=start_point,
                    end=end_point)
                quantized_notes.append(quantized_note)

        quantized_instrument.notes = quantized_notes

        return quantized_instrument

    def quantize(self, filter_res=1 / 16):
        '''Quantize all the midi tracks (instruments), using the predefined
        resolution. The modification is inplace.

        Arg:
        - filter_res: The resolution which indicates the minimun length of
        notes.

        Return:
        - self: The quantized `MidiData` object.
        '''
        return self._process_per_instrument(self.quantize_for_instrument,
                                            filter_res)

    def skyline_for_instrument(self, instrument):
        ''' Apply skyline algorithm to an instrument.

        Arg:
        - instrument: A `pretty_midi.Instrument` object.

        Return:
        - skyline_instrument: A `pretty_midi.Instrument` object with melody
        extracted by skyline algorithm.
        '''

        skyline_instrument = copy.deepcopy(instrument)
        notes = skyline_instrument.notes
        n_notes = len(notes)  # number of notes

        # when the length is too short, don't process it
        if n_notes < 2:
            return skyline_instrument

        # make sure all the notes are sorted by start time
        notes.sort(key=lambda note: note.start)

        # start skyline algorithm
        k = j = 0

        while j < n_notes - 1:
            k = j + 1

            # handle the situation where note_j and note_k have a same pitch
            while (k <= n_notes - 1) and (notes[j].start == notes[k].start):
                if notes[j].pitch < notes[k].pitch:
                    # label note_j's start point -1 for later process
                    notes[j] = pretty_midi.Note(
                        notes[j].velocity, notes[j].pitch, -1, notes[j].end)
                    j = k
                    k += 1
                else:
                    # label note_k's start point -1 for later process
                    notes[k] = pretty_midi.Note(
                        notes[k].velocity, notes[k].pitch, -1, notes[k].end)
                    k += 1

            if k > n_notes - 1:
                break

            # handle the situation where note_j comes before note_k
            if notes[j].end > notes[k].start:
                notes[j] = pretty_midi.Note(notes[j].velocity, notes[j].pitch,
                                            notes[j].start, notes[k].start)

            j = k

        # eliminate all the notes whose start points are -1
        notes = [note for note in notes if note.start != -1]
        skyline_instrument.notes = notes

        return skyline_instrument

    def skyline(self):
        ''' Apply skyline algorithm to all the tracks (instruments).
        The Modification is inplace.

        Return:
        - self: The `MidiData` object that has been applied to skyline.
        '''
        return self._process_per_instrument(self.skyline_for_instrument)

    def note_to_array_for_instrument(self, instrument):
        ''' Convert all the notes in a monophony track to numpy arrays.

        Args:
        - instrument: A `pretty_midi.Instrument` object, which should be
        monophony.

        Return:
        - note_list: A list that contains all the notes, which are converted
        to numpy arrays. The order: (interval, duartion, rest).
        '''

        res = self.res
        beats = self.beats
        notes = instrument.notes

        # convert the unit of resolution from bar to beat.
        res_coef = res * 4

        # WORKAROUND There should be a more general way to do this,
        # but now I just assume the tempo of a whole song is fixed.
        sec_per_beat = float(beats[1] - beats[0])
        sec_unit = sec_per_beat * res_coef  # the time unit for quantization

        # all the elements are numpy arrays
        note_list = []

        last_end = 0
        last_pitch = 0
        for note in notes:
            # TODO make sure use the smallest unit to save space
            # Notice that I have also trimmed the data to prevent overflow,
            # which means if I need to change the dtype, I also have to change
            # the trimming rule.
            # (interval, duartion, rest)
            # Note that in utils.py, function multihot_to_note uses np.int8 too.
            current_array = np.zeros(3, dtype=np.int8)

            # get the duration
            start = note.start
            end = note.end
            duration = round((end - start) / sec_unit)
            # Trim the data to prevent overflow
            if duration >= 128:
                duration = 127
            current_array[1] = duration

            if last_end == 0:
                # get the interval
                interval = 0
                current_array[0] = interval
                # add current note to note_list
                note_list.append(current_array)
            else:
                # get the interval
                interval = note.pitch - last_pitch
                # Trim the data to prevent overflow
                if interval >= 128:
                    interval = 127
                elif interval < -128:
                    interval = -128
                current_array[0] = interval

                # get the last note's rest
                rest = round((note.start - last_end) / sec_unit)
                # Trim the data to prevent overflow
                if rest >= 128:
                    rest = 127
                note_list[-1][2] = rest
                # add current note to note_list
                note_list.append(current_array)

            last_pitch = note.pitch
            last_end = note.end

        return note_list

    def note_to_array(self):
        instrument_list = []
        for instrument in self.get_instruments():
            instrument_list.append(
                self.note_to_array_for_instrument(instrument))
        return instrument_list

    @staticmethod
    def note_list_to_mididata(note_list, start_pitch=60, tempo=100,
                              res=1 / 16):
        ''' Convert a phrase to a `MidiData` object.
        Args:
        - note_list: A list object corresponding to an instrument.
        - start_pitch: The pitch of the first note.
        - time_per_unit: The actual time (represented as second) of one time
        - res: resolution (the minimum beat)
        unit.

        Return:
        - melody: A `MidiData` object.
        '''
        note_list = copy.deepcopy(note_list)

        time_per_unit = 60 * 4 * res / tempo

        # resolution: ticks per quarter note
        # tempo: beats (quarter note) per minute
        pm_melody = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        # TODO The program name can be changed to piano.
        pm_melody_program = pretty_midi.instrument_name_to_program('Cello')
        pm_melody_instrument = pretty_midi.Instrument(
            program=pm_melody_program)

        last_pitch = start_pitch
        last_time_point = 0

        for note in note_list:
            interval = note[0]
            duration = note[1]
            rest = note[2]

            pitch = last_pitch + interval
            duration_time = duration * time_per_unit
            rest_time = rest * time_per_unit

            pm_note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=last_time_point,
                end=last_time_point + duration_time)

            pm_melody_instrument.notes.append(pm_note)
            last_time_point += duration_time + rest_time
            last_pitch = pitch

        pm_melody.instruments.append(pm_melody_instrument)
        melody = MidiData(pm_melody)

        return melody
