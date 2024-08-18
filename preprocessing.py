import numpy as np
import pickle
import re
import os

class Preprocessing:

  FILENAME = "text/spa.txt"
  NUM_DOCS = 20001

  def __init__(self):

    # Building empty lists to hold sentences
    self.input_docs = list()
    self.target_docs = list()

    # Building empty vocabulary sets
    self.input_tokens = set()
    self.target_tokens = set()

    with open(self.FILENAME, 'r', encoding='utf-8') as file:
      lines = file.read().split('\n')

    for line in lines[:self.NUM_DOCS]:
      
      # Input and target sentences are separated by tabs
      input_doc, target_doc = line.split('\t')[:2]
      
      self.input_docs.append(input_doc)

      # Preprocess target sentence (remove punctuation, prefix with <START> and suffix with <END>)
      target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
      target_doc = '<START> ' + target_doc + ' <END>'
      self.target_docs.append(target_doc)

      # Add unique vocabulary token from input and target sentence
      for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        if token not in self.input_tokens:
          self.input_tokens.add(token)
      
      for token in target_doc.split():
        if token not in self.target_tokens:
          self.target_tokens.add(token)

    self.input_tokens = sorted(list(self.input_tokens))
    self.target_tokens = sorted(list(self.target_tokens))

    self.num_encoder_tokens = len(self.input_tokens)
    self.num_decoder_tokens = len(self.target_tokens)

    self.max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in self.input_docs])
    self.max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in self.target_docs])

    self.input_features_dict = { token: i for i, token in enumerate(self.input_tokens) }
    self.target_features_dict = { token: i for i, token in enumerate(self.target_tokens) }

    self.reverse_input_features_dict = { i: token for i, token in enumerate(self.input_tokens) }
    self.reverse_target_features_dict = { i: token for i, token in enumerate(self.target_tokens) }

    self.encoder_input_data = np.zeros(
        (len(self.input_docs), self.max_encoder_seq_length, self.num_encoder_tokens),
        dtype='float32')

    self.decoder_input_data = np.zeros(
        (len(self.input_docs), self.max_decoder_seq_length, self.num_decoder_tokens),
        dtype='float32')

    self.decoder_target_data = np.zeros(
        (len(self.input_docs), self.max_decoder_seq_length, self.num_decoder_tokens),
        dtype='float32')

    # One-hot encoding for input and target sentences
    for line, (input_doc, target_doc) in enumerate(zip(self.input_docs, self.target_docs)):
      
      for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
        self.encoder_input_data[line, timestep, self.input_features_dict[token]] = 1.

      for timestep, token in enumerate(target_doc.split()):
        self.decoder_input_data[line, timestep, self.target_features_dict[token]] = 1.
        
        # Used for teacher forcing

        if timestep > 0:
          self.decoder_target_data[line, timestep - 1, self.target_features_dict[token]] = 1.

if __name__ == "__main__":

  if os.path.exists("preprocessing.pickle"):

    print("Loading pickle file...")

    with open("preprocessing.pickle", "rb") as file:
      obj = pickle.load(file)

  else:

    print("Creating pickle file...")

    obj = Preprocessing()

    with open("preprocessing.pickle", "wb") as file:
      pickle.dump(obj, file)
