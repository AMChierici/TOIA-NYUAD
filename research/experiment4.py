#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:04:44 2019
https://towardsdatascience.com/nlp-sequence-to-sequence-networks-part-1-processing-text-data-d141a5643b72
@author: amc
"""

import pandas as pd
import numpy as np
import string
from string import digits
import re

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, CuDNNLSTM, Input, Embedding, TimeDistributed, Flatten, Dropout
from keras.callbacks import ModelCheckpoint

# read txtfile  :
file_name = 'interview2.csv'
lines = pd.read_csv(file_name, encoding='ISO-8859-1')



# Convert text to lowercase :
lines.Q=lines.Q.apply(lambda x: x.lower())
lines.A=lines.A.apply(lambda x: x.lower())

# Process commas :
lines.Q=lines.Q.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
lines.A=lines.A.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))

# Getting rid of punctuation
exclude = set(string.punctuation)
lines.Q=lines.Q.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines.A=lines.A.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

# Getting rid of digits
remove_digits = str.maketrans('', '', digits)
lines.Q=lines.Q.apply(lambda x: x.translate(remove_digits))
lines.A=lines.A.apply(lambda x: x.translate(remove_digits))



# Appending SOS andEOS to target data : 
lines.A = lines.A.apply(lambda x : 'SOS_ '+ x + ' _EOS')

# Create word dictionaries :
Q_words=set()
for line in lines.Q:
    for word in line.split():
        if word not in Q_words:
            Q_words.add(word)
    
A_words=set()
for line in lines.A:
    for word in line.split():
        if word not in A_words:
            A_words.add(word)
            
# get lengths and sizes :
num_Q_words = len(Q_words)
num_A_words = len(A_words)

max_Q_words_per_sample = max([len(sample.split()) for sample in lines.Q])+5
max_A_words_per_sample = max([len(sample.split()) for sample in lines.A])+5

num_Q_samples = len(lines.Q)
num_A_samples = len(lines.A)

# Get lists of words :
input_words = sorted(list(Q_words))
target_words = sorted(list(A_words))

Q_token_to_int = dict()
Q_int_to_token = dict()

A_token_to_int = dict()
A_int_to_token = dict()

#Tokenizing the words ( Convert them to numbers ) :
for i,token in enumerate(input_words):
    Q_token_to_int[token] = i
    Q_int_to_token[i]     = token

for i,token in enumerate(target_words):
    A_token_to_int[token] = i
    A_int_to_token[i]     = token

# initiate numpy arrays to hold the data that our seq2seq model will use:
encoder_input_data = np.zeros(
    (num_Q_samples, max_Q_words_per_sample),
    dtype='float32')
decoder_input_data = np.zeros(
    (num_A_samples, max_A_words_per_sample),
    dtype='float32')
decoder_target_data = np.zeros(
    (num_A_samples, max_A_words_per_sample, num_A_words),
    dtype='float32')

# Process samples, to get input, output, target data:
for i, (input_text, target_text) in enumerate(zip(lines.Q, lines.A)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = Q_token_to_int[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = A_token_to_int[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, A_token_to_int[word]] = 1.
            
            
            


# Defining some constants: 
vec_len       = 300   # Length of the vector that we willl get from the embedding layer
latent_dim    = 1024  # Hidden layers dimension 
dropout_rate  = 0.2   # Rate of the dropout layers
batch_size    = 64    # Batch size
epochs        = 30    # Number of epochs

# Define an input sequence and process it.
# Input layer of the encoder :
encoder_input = Input(shape=(None,))

# Hidden layers of the encoder :
encoder_embedding = Embedding(input_dim = num_Q_words, output_dim = vec_len)(encoder_input)
encoder_dropout   = (TimeDistributed(Dropout(rate = dropout_rate)))(encoder_embedding)
encoder_LSTM      = LSTM(latent_dim, return_sequences=True)(encoder_dropout)

# Output layer of the encoder :
encoder_LSTM2_layer = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_LSTM2_layer(encoder_LSTM)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]







# Set up the decoder, using `encoder_states` as initial state.
# Input layer of the decoder :
decoder_input = Input(shape=(None,))

# Hidden layers of the decoder :
decoder_embedding_layer = Embedding(input_dim = num_A_words, output_dim = vec_len)
decoder_embedding = decoder_embedding_layer(decoder_input)

decoder_dropout_layer = (TimeDistributed(Dropout(rate = dropout_rate)))
decoder_dropout = decoder_dropout_layer(decoder_embedding)

decoder_LSTM_layer = LSTM(latent_dim, return_sequences=True)
decoder_LSTM = decoder_LSTM_layer(decoder_dropout, initial_state = encoder_states)

decoder_LSTM_2_layer = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_LSTM_2,_,_ = decoder_LSTM_2_layer(decoder_LSTM)

# Output layer of the decoder :
decoder_dense = Dense(num_A_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_LSTM_2)








# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_input, decoder_input], decoder_outputs)

model.summary()

# Define a checkpoint callback :
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]






num_train_samples = 100
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data[:num_train_samples,:],
               decoder_input_data[:num_train_samples,:]],
               decoder_target_data[:num_train_samples,:,:],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.08,
          callbacks = callbacks_list)



utterance = 'do you study here'
response = 'where here'

encoder_predict_data = np.zeros((1, max_Q_words_per_sample), dtype='float32')
decoder_predict_data = np.zeros((1, max_A_words_per_sample), dtype='float32')

for t, word in enumerate(utterance.split()):
    encoder_predict_data[0, t] = Q_token_to_int[word]
    
for t, word in enumerate(response.split()):
        decoder_predict_data[0, t] = A_token_to_int[word]
        
prediction = model.predict([encoder_predict_data, decoder_predict_data])







