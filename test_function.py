from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
from preprocessing import Preprocessing
from training_model import Training
from tensorflow import keras
import numpy as np
import pickle

with open("preprocessing.pickle", "rb") as file:
  preprocessing = pickle.load(file)

with open("training_model.pickle", "rb") as file:
  training = pickle.load(file)

training_model = load_model('training_model.keras')
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_hidden = Input(shape=(training.latent_dim,))
decoder_state_input_cell = Input(shape=(training.latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
decoder_outputs, state_hidden, state_cell = training.decoder_lstm(training.decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = training.decoder_dense(decoder_outputs)
decoder_model = Model([training.decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_sequence(test_input):
  # Encode the input as state vectors.
  states_value = encoder_model.predict(test_input)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1, preprocessing.num_decoder_tokens))
  # Populate the first token of target sequence with the start token.
  target_seq[0, 0, preprocessing.target_features_dict['<START>']] = 1.

  # Sampling loop for a batch of sequences
  # (to simplify, here we assume a batch of size 1).
  decoded_sentence = ''

  stop_condition = False
  while not stop_condition:
    # Run the decoder model to get possible 
    # output tokens (with probabilities) & states
    output_tokens, hidden_state, cell_state = decoder_model.predict(
      [target_seq] + states_value)

    # Choose token with highest probability
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_token = preprocessing.reverse_target_features_dict[sampled_token_index]
    decoded_sentence += " " + sampled_token

    # Exit condition: either hit max length
    # or find stop token.
    if (sampled_token == '<END>' or len(decoded_sentence) > preprocessing.max_decoder_seq_length):
      stop_condition = True

    # Update the target sequence (of length 1).
    target_seq = np.zeros((1, 1, preprocessing.num_decoder_tokens))
    target_seq[0, 0, sampled_token_index] = 1.

    # Update states
    states_value = [hidden_state, cell_state]

  return decoded_sentence


# CHANGE RANGE (NUMBER OF TEST SENTENCES TO TRANSLATE) AS YOU PLEASE
for seq_index in range(100):
  test_input = preprocessing.encoder_input_data[seq_index: seq_index + 1]
  decoded_sentence = decode_sequence(test_input)
  print('-')
  print('Input sentence:', preprocessing.input_docs[seq_index])
  print('Decoded sentence:', decoded_sentence)