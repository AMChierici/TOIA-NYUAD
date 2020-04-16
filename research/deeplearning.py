#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:18:29 2020

@author: amc
"""

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
import tensorflow_text
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


Context, WOzAnswers, Labels= [], [], []
KBanswers = list(np.unique(knowledgebase.Utterance.values))
for example_id in range(len(validation_dialogues)):
    exampleWOzAnswers = list(validation_dialogues.iloc[example_id, 7:].values)
    no_distr_to_sample = 0
    ids_to_exclude = []
    if allnull(exampleWOzAnswers):
        Context.append(validation_dialogues.iloc[example_id, 4])
        WOzAnswers.append("I don't have an answer for this question")
        Labels.append(1)
        Context.append(validation_dialogues.iloc[example_id, 4])
        np.random.seed(example_id)
        WOzAnswers.append(str(np.random.choice(KBanswers, 1)[0]))
        Labels.append(0)
    else:  
        for answer in exampleWOzAnswers:
            if not pd.isnull(answer):
                Context.append(validation_dialogues.iloc[example_id, 4])
                WOzAnswers.append(answer)
                Labels.append(1)
                ids_to_exclude.append(KBanswers.index(answer))
                no_distr_to_sample += 1      
        #
        tmp_distractors = [KBanswers[i] for i in
                np.array(range(len(KBanswers)))
                [np.isin(range(len(KBanswers)), ids_to_exclude, invert=True)]
                ]
        #
        np.random.seed(example_id)
        if no_distr_to_sample==1:
            np.random.seed(example_id)
            answer = str(np.random.choice(tmp_distractors, 1))
        else:    
            for answer in np.random.choice(tmp_distractors, no_distr_to_sample, replace=False):
                Context.append(validation_dialogues.iloc[example_id, 4])
                WOzAnswers.append(str(answer))
                Labels.append(0)
                
valid_df = pd.DataFrame({'Context':Context, 'Utterance':WOzAnswers, 'Label':Labels})
for i in range(len(valid_df)):
    try:
        tmp_df = pd.DataFrame({
            'Context': eda(valid_df.Context.values[i], .2, .2, .2, .2, 15),
            'Utterance': list(np.repeat(valid_df.Utterance.values[i], 16)),
            'Label': list(np.repeat(valid_df.Label.values[i], 16))
            })
        valid_df = valid_df.append(tmp_df)
    except Exception:
        pass

valid_df.reset_index(level=None, drop=True, inplace=True)
  

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True
      
      
stopwords = ['', 'a', 'about', 'above', 'after', 'again', 'against', 'all',
       'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because',
       'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
       'can', 'could', 'did', 'do', 'does', 'doing', 'don', 'down',
       'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has',
       'have', 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',
       "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how',
       "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into',
       'is', 'it', "it's", 'its', 'itself', 'just', "let's", 'me', 'more',
       'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off',
       'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
       'ourselves', 'out', 'over', 'own', 's', 'same', 'she', "she'd",
       "she'll", "she's", 'should', 'so', 'some', 'such', 't', 'than',
       'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves',
       'then', 'there', "there's", 'these', 'they', "they'd", "they'll",
       "they're", "they've", 'this', 'those', 'through', 'to', 'too',
       'under', 'until', 'up', 'very', 'was', 'we', "we'd", "we'll",
       "we're", "we've", 'were', 'what', "what's", 'when', "when's",
       'where', "where's", 'which', 'while', 'who', "who's", 'whom',
       'why', "why's", 'will', 'with', 'would', 'you', "you'd", "you'll",
       "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
print(len(stopwords))
# Expected Output
# 164
 
training_corpus = [] 
for question, answer in zip(train_df.Context.values, train_df.Utterance.values):
    sentence = question.lower() + " " + answer.lower()
    for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
    training_corpus.append(sentence)
    
validation_corpus = []
for question, answer in zip(valid_df.Context.values, valid_df.Utterance.values):
    sentence = question.lower() + " " + answer.lower()
    for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
    validation_corpus.append(sentence)

vocab_size = 4000
embedding_dim = 512
max_length = 120
trunc_type='post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_corpus)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_corpus)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

validation_sequences = tokenizer.texts_to_sequences(validation_corpus)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

print(decode_qapair(padded[3]))
print(training_corpus[3])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 20
callbacks = myCallback()
history= model.fit(padded, train_df.Label.values, epochs=num_epochs, validation_data=(validation_padded, valid_df.Label.values), callbacks=[callbacks])

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

y_valid = valid_df.Label.values
y_lstm_probs = model.predict(validation_padded)
y_lstm = [1 if i>.5 else 0 for i in y_lstm_probs]
TN, FP, FN, TP = confusion_matrix(y_valid, y_lstm, labels=[1,0]).ravel()
thr = TP/(2*TP + FN + FP)
print("\n CM: ", confusion_matrix(y_valid, y_lstm, labels=[1,0]),
    "\n Accuracy: ", accuracy_score(y_valid, y_lstm),
    "\n Precision: ", precision_score(y_valid, y_lstm),
    "\n Recall: ", recall_score(y_valid, y_lstm),
    "\n F1: ", f1_score(y_valid, y_lstm),
)


input_text = "Does NYU Abu Dhabi take into consideration the difficulty of a particular curriculum or different secondary school grading policies when making admissions decisions?"

predictions = []
for A in KBanswers:
    sentence = input_text + " " + A
    for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
    predictions.append(sentence)

sequences = tokenizer.texts_to_sequences(predictions)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
   
k=10
rankings = model.predict(padded)

print("Question: ", input_text)
for answer, ranking in zip(np.take(KBanswers, list(np.argsort(rankings, axis=0)[::-1][:k])), np.sort(rankings, axis=0)[::-1][:k]):
    print(
          "\n Answer: ", answer[0],
          "\n [Rank value: ", ranking*1000, "]"
      )
    
    
# Now using pre-trained embeddings:

# questions = ["What is your age?"]
# responses = ["I am 20 years old.", "good morning"]
# response_contexts = ["I will be 21 next year.", "great day."]
##Normally context are sentences before/after the answer.

module = hub.load('./3') #https://tfhub.dev/google/universal-sentence-encoder-qa/3 ##downlaod at https://tfhub.dev/google/universal-sentence-encoder-qa/3

# question_embeddings = module.signatures['question_encoder'](
#             tf.constant(questions))
# response_embeddings = module.signatures['response_encoder'](
#         input=tf.constant(responses),
#         context=tf.constant(response_contexts))

# np.inner(question_embeddings, response_embeddings)

training_corpus = [] 
for question, answer in zip(train_df.Context.values, train_df.Utterance.values):
    sentence = question.lower() + " " + answer.lower()
    training_corpus.append(sentence)
training_embeddings = module.signatures['question_encoder'](
            tf.constant(training_corpus))
training_embeddings=np.array(training_embeddings['outputs'])
    
validation_corpus = []
for question, answer in zip(valid_df.Context.values, valid_df.Utterance.values):
    sentence = question.lower() + " " + answer.lower()
    validation_corpus.append(sentence)
validation_embeddings = module.signatures['question_encoder'](
            tf.constant(validation_corpus))
validation_embeddings=np.array(validation_embeddings['outputs'])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build([29880, 512])
model.summary()

num_epochs = 100
# callbacks = myCallback()
callbacks = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history= model.fit(training_embeddings, train_df.Label.values, epochs=num_epochs, validation_data=(validation_embeddings, valid_df.Label.values), callbacks=[callbacks])
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

y_valid = valid_df.Label.values
y_nnet_probs = model.predict(validation_embeddings)
y_nnet = [1 if i>.5 else 0 for i in y_nnet_probs]
TN, FP, FN, TP = confusion_matrix(y_valid, y_nnet, labels=[1,0]).ravel()
print("\n CM: ", np.array([[TP, FN], [FP, TN]]),
    "\n Accuracy: ", accuracy_score(y_valid, y_nnet),
    "\n Precision: ", precision_score(y_valid, y_nnet),
    "\n Recall: ", recall_score(y_valid, y_nnet),
    "\n F1: ", f1_score(y_valid, y_nnet),genade.co
)
thr = TP/(2*TP + FN + FP)
y_nnet = [1 if i>thr else 0 for i in y_nnet_probs]
print("\n CM: ", confusion_matrix(y_valid, y_nnet),
    "\n Accuracy: ", accuracy_score(y_valid, y_nnet),
    "\n Precision: ", precision_score(y_valid, y_nnet),
    "\n Recall: ", recall_score(y_valid, y_nnet),
    "\n F1: ", f1_score(y_valid, y_nnet),
)

input_text = "Where are you from?"

predictions = []
for A in KBanswers:
    sentence = input_text + " " + A
    predictions.append(sentence)
embeddings = module.signatures['question_encoder'](
            tf.constant(predictions))
embeddings=np.array(embeddings['outputs'])

k=50
rankings = model.predict(embeddings)

print("Question: ", input_text)
for answer, ranking in zip(np.take(KBanswers, list(np.argsort(rankings, axis=0)[::-1][:k])), np.sort(rankings, axis=0)[::-1][:k]):
    print(
          "\n Answer: ", answer[0],
          "\n [Rank value: ", ranking*1000, "]"
      )

pred_answers = np.take(KBanswers, list(np.argsort(rankings, axis=0)[::-1][:k]))
pred_answers = [answer[0] for answer in pred_answers]
step1_corpus = list(knowledgebase.loc[knowledgebase.Utterance.isin(pred_answers), 'Context'])
tokenized_corpus = [doc.split(" ") for doc in step1_corpus]

k=5
bm25 = BM25Okapi(tokenized_corpus)
step2_rankings = bm25.get_scores(input_text.split(" "))
step2_ranked_questions = np.take(step1_corpus, list(np.argsort(step2_rankings, axis=0)[::-1][:k]))
step2_answers = [knowledgebase.loc[knowledgebase.Context == a, "Utterance"] for a in step2_ranked_questions]

print("Reranked answer: ", step2_answers[0])