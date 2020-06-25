#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:50:44 2020

@author: amc
"""

# MIT License
#
# Copyright 2018-2019 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
nltk.download('wordnet')
from eda import *
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import *
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from rank_bm25 import BM25Okapi
import matplotlib.pyplot as plt
import io
from helper_functions import *

def allnull(somelist):
    count=0
    for i in somelist:
        if pd.isnull(i):
            count+=1
    return count==len(somelist)

def decode_qapair(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
knowledgebase = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/MargaritaCorpusKB.csv', encoding='utf-8')
train_test_dialogues = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/DIALOGUES.csv', encoding='utf-8')

validation_dialogues = train_test_dialogues.loc[train_test_dialogues.Experiment=="TRAIN"]
test_dialogues = train_test_dialogues.loc[train_test_dialogues.Experiment=="TEST"]

KBanswers = list(np.unique(knowledgebase.Utterance.values))

Context, Utterance, Labels= [], [], []
for example_id in range(len(knowledgebase)):
    no_distr_to_sample = 0
    Context.append(knowledgebase.Context.values[example_id])
    Utterance.append(knowledgebase.Utterance.values[example_id])
    Labels.append(1)
    id_to_exclude = KBanswers.index(knowledgebase.Utterance.values[example_id])
    tmp_distractors = [KBanswers[i] for i in
            np.array(range(len(KBanswers)))
            [np.isin(range(len(KBanswers)), id_to_exclude, invert=True)]
            ]
    np.random.seed(example_id)
    Context.append(knowledgebase.Context.values[example_id])
    Utterance.append(np.random.choice(tmp_distractors, 1)[0])
    Labels.append(0)

train_df = pd.DataFrame({'Context':Context, 'Utterance':Utterance, 'Label':Labels})

for i in range(len(train_df)):
    try:
        tmp_df = pd.DataFrame({
            'Context': eda(train_df.Context.values[i], .2, .2, .2, .2, 15),
            'Utterance': list(np.repeat(train_df.Utterance.values[i], 16)),
            'Label': list(np.repeat(train_df.Label.values[i], 16))
            })
        train_df = train_df.append(tmp_df)
    except Exception:
        pass

train_df.reset_index(level=None, drop=True, inplace=True)
###################
ar_msk = np.random.rand(len(train_df)) < 2/3
train_df[ar_msk].to_csv('data/data.csv', encoding='utf-8', index=False)
train_df[~ar_msk].to_csv('data/data_test.csv', encoding='utf-8', index=False)
###-clear space-###
del(train_df, ar_msk)
###################

import os
import bert

__file__='/Users/amc/Documents/TOIA-NYUAD/research/'

def createTokenizer():
    currentDir = os.path.dirname(os.path.realpath(__file__))
    modelsFolder = os.path.join(currentDir, 'research/models', 'uncased_L-2_H-128_A-2')
    vocab_file = os.path.join(modelsFolder, 'vocab.txt')

    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer

tokenizer = createTokenizer()

import csv
import random

dataDir = 'data'
max_seq_length = 256

def loadData(tokenizer):
    fileName = os.path.join(dataDir, "data.csv")
    fileTestName = os.path.join(dataDir, "data_test.csv")

    data = []
    data_test = []
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []

    with open(fileName, encoding='utf-8') as csvFile:
        csv_reader = csv.reader(csvFile)
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                data.append(row) 
            line_count +=1
    csvFile.close()

    with open(fileTestName, encoding='utf-8') as csvFileTest:
        csv_reader_test = csv.reader(csvFileTest)
        line_count = 0
        for row in csv_reader_test:
            if line_count > 0:
                data_test.append(row)
            line_count +=1
    csvFileTest.close()

    shuffled_set = random.sample(data, len(data))
    training_set = shuffled_set[0:]
    shuffled_set_test = random.sample(data_test, len(data_test))
    testing_set = shuffled_set_test[0:]

    for el in training_set:
        train_set.append(el[0] + el[1])
        train_labels.append(float(el[2]))

    for el in testing_set:
        test_set.append(el[0] + el[1])
        test_labels.append(float(el[2]))

    #defineTokenizerConfig(train_set)

    train_tokens = map(tokenizer.tokenize, train_set)
    train_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], train_tokens)
    train_token_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))

    train_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), train_token_ids)
    train_token_ids = np.array(list(train_token_ids))

    test_tokens = map(tokenizer.tokenize, test_set)
    test_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], test_tokens)
    test_token_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))

    test_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), test_token_ids)
    test_token_ids = np.array(list(test_token_ids))

    train_labels_final = np.array(train_labels)
    test_labels_final = np.array(test_labels)

    return train_token_ids, train_labels_final, test_token_ids, test_labels_final

train_set, train_labels, test_set, test_labels = loadData(tokenizer)


import bert
import os

def createBertLayer():
    global bert_layer
    bertDir = os.path.join(modelBertDir, "uncased_L-2_H-128_A-2")
    bert_params = bert.params_from_pretrained_ckpt(bertDir)
    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
    bert_layer.apply_adapter_freeze()

modelBertDir='models'
createBertLayer()

def loadBertCheckpoint():
    modelsFolder = os.path.join(modelBertDir, "uncased_L-2_H-128_A-2")
    checkpointName = os.path.join(modelsFolder, "bert_model.ckpt")
    bert.load_stock_weights(bert_layer, checkpointName)

    
import tensorflow as tf

def createModel():
    global model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='input_ids'),
        bert_layer,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(None, max_seq_length))
    print(model.summary())

createModel()

def fitModel(training_set, training_label, testing_set, testing_label):
    global history
    checkpointName = os.path.join(modelDir, "bert_faq.ckpt")
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointName,
                                                     save_weights_only=True,
                                                     verbose=1)
    # callback = StopTrainingClassComplete()
    history = model.fit(
        training_set,
        training_label,
        epochs=10,
        batch_size=int(len(training_set)/32),
        validation_data=(testing_set, testing_label),
        verbose=1,
        callbacks=[cp_callback]
    )

modelDir='models'
fitModel(train_set, train_labels, test_set, test_labels)

import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
