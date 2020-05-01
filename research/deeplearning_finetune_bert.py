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

df_knowledgebase = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/MargaritaCorpusKB.csv', encoding='utf-8')
#take random subset, only Q and A
df_dataset = df_knowledgebase[['Context', 'Utterance']].sample(n=50, random_state=1)
###-clear space-###
del(df_knowledgebase)
###################
df_dataset = df_dataset.rename(columns={"Context": "SENTENCE"})
#create lookup table. Utterances will become labels
ls_utterances = list(np.unique(df_dataset.Utterance.values))
df_lookup = pd.DataFrame({'ID':1 + np.array(range(len(ls_utterances))), 'Utterance':ls_utterances})
###-clear space-###
del(ls_utterances)
###################
df_dataset=df_dataset.join(df_lookup.set_index('Utterance'), on='Utterance')[['ID', 'SENTENCE']]
df_dataset.reset_index(level=None, drop=True, inplace=True)

df_augmented_dataset = pd.DataFrame({'ID':[], 'SENTENCE':[]})
from eda import *
for i in range(len(df_dataset)):
    try:
        df_singlesentence = pd.DataFrame({
            'ID': list(np.repeat(df_dataset.ID.values[i], 16)),
            'SENTENCE': eda(df_dataset.SENTENCE.values[i], .2, .2, .2, .2, 15)
            })
        df_augmented_dataset = df_augmented_dataset.append(df_singlesentence)
    except Exception:
        pass
df_augmented_dataset.reset_index(level=None, drop=True, inplace=True)
###-clear space-###
del(i, df_singlesentence, stop_words, wordnet)
###################
ar_msk = np.random.rand(len(df_augmented_dataset)) < 2/3
df_augmented_dataset[ar_msk].to_csv('data/data.csv', encoding='utf-8', index=False)
df_augmented_dataset[~ar_msk].to_csv('data/data_test.csv', encoding='utf-8', index=False)
###-clear space-###
del(df_dataset, df_augmented_dataset, ar_msk)
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
classes = len(df_lookup)
max_seq_length = 64

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
        train_set.append(el[1])
        zeros = [0] * classes
        zeros[int(float(el[0])) - 1] = 1
        train_labels.append(zeros)

    for el in testing_set:
        test_set.append(el[1])
        zeros = [0] * classes
        zeros[int(float(el[0])) - 1] = 1
        test_labels.append(zeros)

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
        tf.keras.layers.Dense(classes, activation=tf.nn.softmax)
    ])

    model.build(input_shape=(None, max_seq_length))

    model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.0001), metrics=['accuracy'])

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
        epochs=100,
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
