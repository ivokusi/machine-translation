from keras.layers import Input, LSTM, Dense # type: ignore
from keras.models import Model # type: ignore
from preprocessing import Preprocessing
from tensorflow import keras
import pickle
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Training:

    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 50
    EPOCHS = 50

    def __init__(self):

        with open("preprocessing.pickle", "rb") as file:
            preprocessing = pickle.load(file)

        self.latent_dim = 256

        # Encoder training setup
        self.encoder_inputs = Input(shape=(None, preprocessing.num_encoder_tokens), name="encoder_input")
        encoder_lstm = LSTM(self.latent_dim, return_state=True, name="encoder_lstm")
        encoder_outputs, state_hidden, state_cell = encoder_lstm(self.encoder_inputs)
        self.encoder_states = [state_hidden, state_cell]

        # Decoder training setup:
        self.decoder_inputs = Input(shape=(None, preprocessing.num_decoder_tokens), name="decoder_input")
        self.decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
        decoder_outputs, decoder_state_hidden, decoder_state_cell = self.decoder_lstm(self.decoder_inputs, initial_state=self.encoder_states)
        self.decoder_dense = Dense(preprocessing.num_decoder_tokens, activation='softmax')
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # Building the training model:
        training_model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)

        print("Model summary:\n")
        training_model.summary()
        print("\n\n")

        # Compile, train and save model
        training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        training_model.fit([preprocessing.encoder_input_data, preprocessing.decoder_input_data], preprocessing.decoder_target_data, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, validation_split=self.VALIDATION_SPLIT)
        training_model.save('training_model.keras')

if __name__ == "__main__":

  if os.path.exists("training_model.pickle"):

    print("Loading pickle file...")

    with open("training_model.pickle", "rb") as file:
      obj = pickle.load(file)

  else:

    print("Creating pickle file...")

    obj = Training()

    with open("training_model.pickle", "wb") as file:
      pickle.dump(obj, file)
