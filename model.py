# -*-coding: utf-8-*-
# Author: Yilin Zhang

# External imports
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Dropout
from keras.optimizers import Adadelta
from keras.callbacks import TensorBoard, ModelCheckpoint

# Internal imports
from utils import gen_batch

# import constants
from configs import MODEL_PATH, LOG_PATH, LENGTH_LIMIT
from configs import INTERVAL_RANGE, DURATION_RANGE, REST_RANGE

# Fixed model parameters
n_steps = LENGTH_LIMIT
n_inputs = INTERVAL_RANGE + DURATION_RANGE + REST_RANGE
n_outputs = DURATION_RANGE + REST_RANGE

# Hyper parameters
n_neurons = 256


class XRModel():
    def construct(self, dropout_rate):
        ''' Construct the model.
        - dropout_rate: Dropout rate.
        '''
        model_input = Input(batch_shape=(None, n_steps, n_inputs))
        x = LSTM(
            n_neurons, activation='elu', return_sequences=True)(model_input)
        x = Dropout(dropout_rate)(x)
        x = LSTM(n_neurons, activation='elu', return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)
        x = LSTM(n_neurons, activation='elu', return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)
        duration_output = Dense(
            DURATION_RANGE, activation='softmax', name='duration_output')(x)
        rest_output = Dense(
            REST_RANGE, activation='softmax', name='rest_output')(x)
        self.model = Model(
            inputs=model_input, outputs=[duration_output, rest_output])

    def load(self, model_path, dropout_rate):
        ''' Load the pre-trained model. (You can change the dropout rate here.)
        - model_path: The pre-trained model's path.
        - dropout_rate: Dropout rate.
        '''
        self.construct(dropout_rate)
        model = load_model(model_path)
        # Copy all the weights to the new model
        for new_layer, layer in zip(self.model.layers[1:], model.layers[1:]):
            new_layer.set_weights(layer.get_weights())

    def compile(self):
        ''' Compile the model.
        '''
        self.model.compile(
            optimizer=Adadelta(),
            loss={
                'duration_output': 'categorical_crossentropy',
                'rest_output': 'categorical_crossentropy'
            },
            loss_weights={
                'duration_output': 0.5,
                'rest_output': 0.5
            },
            metrics=['accuracy'])

    def fit(self,
            train_path,
            valid_path,
            batch_size,
            steps_per_epoch,
            validation_steps,
            n_epochs,
            initial_epoch=0):
        ''' Fit the model.
        - trian_path: The training set's path.
        - valid_path: The validation set's path.
        - bath_size: The batch size.
        - steps_per_epoch: The steps you need to go through a whole training
        set.
        - validation_steps: The steps you need to go through a whole validation
        set.
        - n_epochs: How many epochs do you want.
        - initial_epoch: It depends on your pre-trained model.
        '''

        # Set tensorboard callback
        tb_callback = TensorBoard(log_dir=LOG_PATH, batch_size=batch_size)
        model_save_path = MODEL_PATH + "saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
        mc_callback = ModelCheckpoint(
            filepath=model_save_path, monitor='val_loss')

        # Summary the model
        self.model.summary()

        # Fit model
        gen_train = gen_batch(train_path, n_steps, batch_size)
        gen_valid = gen_batch(valid_path, n_steps, batch_size)
        self.model.fit_generator(
            gen_train,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
            validation_data=gen_valid,
            validation_steps=validation_steps,
            callbacks=[tb_callback, mc_callback],
            workers=2,
            use_multiprocessing=True,
            initial_epoch=initial_epoch)
