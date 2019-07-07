#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:03:53 2019

@author: amc
"""

import numpy as np
import pandas as pd
import create_datasets
from create_datasets import *
#from create_datasets_nonullWOz import *
import time


## Baseline using ooc data as training ##
### No pre-processing ###

train_df = zeroturndialogue.rename(columns = {"Question":"Context", "Answer": "Utterance"})

hold_out_sample = np.array([3, 7])
conversation_IDs = np.unique(multiturndialogues.ConversationID)
valid_samples = conversation_IDs[np.isin(conversation_IDs, hold_out_sample, invert=True)]
test_samples = conversation_IDs[np.isin(conversation_IDs, hold_out_sample)]

valid_df = multiturndialogues.loc[multiturndialogues.ConversationID.isin(valid_samples)]
valid_df.reset_index(level=None, drop=True, inplace=True)
valid_df = test_set_questions_ooctrain(valid_df)

test_df = multiturndialogues.loc[multiturndialogues.ConversationID.isin(test_samples)]
test_df.reset_index(level=None, drop=True, inplace=True)        
test_df = test_set_questions_ooctrain(test_df)

# Train TFIDF predictor
pred = TFIDFPredictor()
pred.train(train_df)

# Evaluate TFIDF predictor
y = [pred.predict(valid_df.Context[x], list(train_df.Utterance.values)) for x in range(len(valid_df))]
for thr in np.arange(0.0, 1.0, 0.05):
    print("Recall@1 for thr={}: {:g}".format(thr, evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[2]))

# Test TFIDF predictor
y = [pred.predict(test_df.Context[x], list(train_df.Utterance.values)) for x in range(len(test_df))]
for n in [1, 2, 5, 10, 20]:
    print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, test_df.WOzAnswers.values, n, thr=0.25)[0]))

print(
      sum([len(ls)==0 for ls in valid_df.WOzAnswers.values])/len(valid_df.WOzAnswers.values),
      sum([len(ls)==0 for ls in valid_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in valid_df.WOzAnswers.values]) 
      )
print(
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values])/len(test_df.WOzAnswers.values),
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in test_df.WOzAnswers.values]) 
      ) 

#### Adding turns -No pre-process ####
#train_df, test_df = train_test_sets_seqdata(TURNS=1)
#y_test = []
#for i in test_df.WOzAnswers.values:
#    y_test.append(list(range(len(i))))
#pred = TFIDFPredictor()
#pred.train(train_df)
#y_tfifd = [pred.predict(test_df.Context[x], test_df.iloc[x, 1]+test_df.iloc[x, 2]) for x in range(len(test_df))]
#for n in [1, 3, 5, 10, 20]:
#    print("Recall @ ({}): {:g}".format(n, evaluate_recall_modified(y_tfifd, y_test, n)))


#### With pre-processing ###
for i in range(len(train_df)-1):
    train_df.iloc[i, 1] = preprocess(train_df.iloc[i, 1])
    train_df.iloc[i, 2] = preprocess(train_df.iloc[i, 2])
for i in range(len(valid_df)-1):
    valid_df.iloc[i, 0] = preprocess(valid_df.iloc[i, 0])
    valid_df.iloc[i, 1] = [preprocess(utt) for utt in valid_df.iloc[i, 1]]
    valid_df.iloc[i, 2] = [preprocess(utt) for utt in valid_df.iloc[i, 2]]
for i in range(len(test_df)-1):
    test_df.iloc[i, 0] = preprocess(test_df.iloc[i, 0])
    test_df.iloc[i, 1] = [preprocess(utt) for utt in test_df.iloc[i, 1]]
    test_df.iloc[i, 2] = [preprocess(utt) for utt in test_df.iloc[i, 2]]

y_valid = []
for i in valid_df.WOzAnswers.values:
    y_valid.append(list(range(len(i))))
y_test = []
for i in test_df.WOzAnswers.values:
    y_test.append(list(range(len(i))))
    
# Train TFIDF predictor
pred = TFIDFPredictor()
pred.train(train_df)

# Evaluate TFIDF predictor
y = [pred.predict(valid_df.Context[x], valid_df.iloc[x, 1]+valid_df.iloc[x, 2]) for x in range(len(valid_df))]
for thr in np.arange(0.0, 1.0, 0.05):
    print("Recall@1 for thr={}: {:g}".format(thr, evaluate_recall_thr(y, y_valid, k=10, thr=thr)[0]))

# Test TFIDF predictor
y = [pred.predict(test_df.Context[x], test_df.iloc[x, 1]+test_df.iloc[x, 2]) for x in range(len(test_df))]
for n in [1, 2, 5, 10, 20]:
    print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, y_test, n, thr=0.15)[0]))
    



############# Using infersent #############
### No pre-processing ###
zeroturndialogue = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/ooc_data.csv', encoding='utf-8')
multiturndialogues = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/newconversations_woz.csv', encoding='utf-8')  
multiturndialogues_no_ooc_used = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/newconversations_woz_onlynewdata.csv', encoding='utf-8')    

train_df = zeroturndialogue.rename(columns = {"Question":"Context", "Answer": "Utterance"})

hold_out_sample = np.array([3, 7])
conversation_IDs = np.unique(multiturndialogues.ConversationID)
valid_samples = conversation_IDs[np.isin(conversation_IDs, hold_out_sample, invert=True)]
test_samples = conversation_IDs[np.isin(conversation_IDs, hold_out_sample)]

valid_df = multiturndialogues.loc[multiturndialogues.ConversationID.isin(valid_samples)]
valid_df.reset_index(level=None, drop=True, inplace=True)
valid_df = test_set_questions_ooctrain(valid_df)

test_df = multiturndialogues.loc[multiturndialogues.ConversationID.isin(test_samples)]
test_df.reset_index(level=None, drop=True, inplace=True)      
test_df = test_set_questions_ooctrain(test_df)
    
import torch
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
# Load model
from InferSent.encoder.models import InferSent

model_version = 1
MODEL_PATH = "InferSent/encoder/infersent%s.pickle" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'InferSent/dataset/GloVe/glove.840B.300d.txt' if model_version == 1 else 'InferSent/dataset/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

allsentences = []
for item in train_df.Context.values: allsentences.append(item)
for item in train_df.Utterance.values: allsentences.append(item)
#for item in zeroturndialogue.Answer.values: allsentences.append(item)
model.build_vocab(allsentences, tokenize=True)

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def INFERSENTPredictor(context, utterances):
    # The dot product measures the similarity of the resulting vectors
    result = [cosine(model.encode([context])[0], model.encode([utt])[0]) for utt in utterances]
    # Sort by top results and return the indices in descending order
#    return np.argsort(result, axis=0)[::-1]
    return result

train_embeddings = model.encode(train_df.Utterance.values, bsize=128, tokenize=False, verbose=True)
valid_embeddings = model.encode(valid_df.Context.values, bsize=128, tokenize=False, verbose=True)
test_embeddings = model.encode(test_df.Context.values, bsize=128, tokenize=False, verbose=True)

np.argsort([cosine(valid_embeddings[15], utt) for utt in train_embeddings], axis=0)[::-1][:5]
np.sort([cosine(valid_embeddings[15], utt) for utt in train_embeddings], axis=0)[::-1][:5]
valid_df.Context[15]
valid_df.WOzAnswers[15]
train_df.Utterance[238]

def INFERSENTPredictor_new(context, utterances):
    # The dot product measures the similarity of the resulting vectors
    result = [cosine(context, utt) for utt in utterances]
    # Sort by top results and return the indices in descending order
#    return np.argsort(result, axis=0)[::-1]
    return result

# Evaluate InferSent predictor
y = [INFERSENTPredictor_new(valid_embeddings[x], train_embeddings) for x in range(len(valid_embeddings))]
for thr in np.arange(0, 1.0, 0.05):
    print("Recall@1 for thr={}: {:g}".format(thr, evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[2]))

# Test InferSent predictor
y = [INFERSENTPredictor_new(test_embeddings[x], train_embeddings) for x in range(len(test_embeddings))]
for n in [1, 2, 5, 10, 20]:
    print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, test_df.WOzAnswers.values, n, thr=0.7)[0]))
    
print(
      sum([len(ls)==0 for ls in valid_df.WOzAnswers.values])/len(valid_df.WOzAnswers.values),
      sum([len(ls)==0 for ls in valid_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in valid_df.WOzAnswers.values]) 
      )
print(
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values])/len(test_df.WOzAnswers.values),
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in test_df.WOzAnswers.values]) 
      )  
#    
#from joblib import Parallel, delayed
#import multiprocessing
#     
#num_cores = multiprocessing.cpu_count()
#
## Evaluate InferSent predictor
#start_time = time.time()
#y1 = Parallel(n_jobs=num_cores)(delayed(INFERSENTPredictor)(valid_df.Context[x], valid_df.iloc[x, 1]+valid_df.iloc[x, 2]) for x in range(len(valid_df)))
#elapsed_time = time.time() - start_time
#print(elapsed_time/60/60)
#
#for thr in np.arange(0.0, 1.0, 0.05):
#    print("Recall@1 for thr={}: {:g}".format(thr, evaluate_recall_thr(y1, y_valid, k=1, thr=thr)))
#
## Test InferSent predictor
#start_time = time.time()
#y2 = Parallel(n_jobs=num_cores)(delayed(INFERSENTPredictor)(test_df.Context[x], test_df.iloc[x, 1]+test_df.iloc[x, 2]) for x in range(len(test_df)))
#elapsed_time = time.time() - start_time
#print(elapsed_time/60)
#for n in [1, 2, 5, 10, 20]:
#    print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y1, y_test, n, thr=0.9)))

#y[10]
#test_df.iloc[10, 0]
#(test_df.iloc[10, 1]+test_df.iloc[10, 2])[112]
#test_df.iloc[10, 77]
#test_df.iloc[10, 58]
#y_tfifd[10]
    
    
    
######## use only questions from ooc data ###########


zeroturndialogue = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/ooc_data.csv', encoding='ISO-8859-1')
multiturndialogues = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/newconversations_woz.csv', encoding='ISO-8859-1')      

train_df = zeroturndialogue.rename(columns = {"Question":"Context", "Answer": "Utterance"})

hold_out_sample = np.array([3, 7])
conversation_IDs = np.unique(multiturndialogues.ConversationID)
valid_samples = conversation_IDs[np.isin(conversation_IDs, hold_out_sample, invert=True)]
test_samples = conversation_IDs[np.isin(conversation_IDs, hold_out_sample)]

valid_df = multiturndialogues.loc[multiturndialogues.ConversationID.isin(valid_samples)]
valid_df.reset_index(level=None, drop=True, inplace=True)
valid_df = test_set_questions_ooctrain(valid_df)

test_df = multiturndialogues.loc[multiturndialogues.ConversationID.isin(test_samples)]
test_df.reset_index(level=None, drop=True, inplace=True)      
test_df = test_set_questions_ooctrain(test_df)
    
model_version = 1
MODEL_PATH = "InferSent/encoder/infersent%s.pickle" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'InferSent/dataset/GloVe/glove.840B.300d.txt' if model_version == 1 else 'InferSent/dataset/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

allsentences = []
for item in train_df.Context.values: allsentences.append(item)
#for item in train_df.Utterance.values: allsentences.append(item) # used answer too to expand vocabulary
#for item in zeroturndialogue.Answer.values: allsentences.append(item)
model.build_vocab(allsentences, tokenize=True)


train_embeddings = model.encode(train_df.Context.values, bsize=128, tokenize=False, verbose=True)
valid_embeddings = model.encode(valid_df.Context.values, bsize=128, tokenize=False, verbose=True)
test_embeddings = model.encode(test_df.Context.values, bsize=128, tokenize=False, verbose=True)

np.argsort([cosine(valid_embeddings[15], utt) for utt in train_embeddings], axis=0)[::-1][:5]
np.sort([cosine(valid_embeddings[15], utt) for utt in train_embeddings], axis=0)[::-1][:5]
valid_df.Context[15]
train_df.Context[53]


def INFERSENTPredictor_new(context, utterances):
    # The dot product measures the similarity of the resulting vectors
    result = [cosine(context, utt) for utt in utterances]
    # Sort by top results and return the indices in descending order
#    return np.argsort(result, axis=0)[::-1]
    return result

# Evaluate InferSent predictor
y = [INFERSENTPredictor_new(valid_embeddings[x], train_embeddings) for x in range(len(valid_embeddings))]
for thr in np.arange(0, 1.0, 0.05):
    print("Recall@1 for thr={}: {:g}".format(thr, evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[2]))

# Test InferSent predictor
y = [INFERSENTPredictor_new(test_embeddings[x], train_embeddings) for x in range(len(test_embeddings))]
for n in [1, 2, 5, 10, 20]:
    print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, test_df.WOzAnswers.values, n, thr=0.75)[0]))
   
    
    
    
######## Data Setup 2. ###########
    #to try more turns adj the parameter TURNS
train_df, valid_df, test_df = train_test_sets_questions_seqdata_no_ooc(TURNS=0)

##to add redundancy to train:
#train_1t, _, _ = train_test_sets_questions_seqdata_no_ooc(TURNS=1)
#train_2t, _, _ = train_test_sets_questions_seqdata_no_ooc(TURNS=2)
#train_3t, _, _ = train_test_sets_questions_seqdata_no_ooc(TURNS=3)
#train_df = train_df.append(train_1t, ignore_index = True)  
#train_df = train_df.append(train_2t, ignore_index = True)  
#train_df = train_df.append(train_3t, ignore_index = True)  

### TFIDF ###
# Train TFIDF predictor
pred = TFIDFPredictor()
pred.train(train_df)

# Evaluate TFIDF predictor
y = [pred.predict(valid_df.Context[x], list(train_df.Utterance.values)) for x in range(len(valid_df))]
for thr in np.arange(0.0, 1.0, 0.05):
    print("Recall@10 for thr={}: {:g}".format(thr, evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=10, thr=thr)[0]))

# Test TFIDF predictor
y = [pred.predict(test_df.Context[x], list(train_df.Utterance.values)) for x in range(len(test_df))]
for n in [1, 2, 5, 10, 20]:
    print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, test_df.WOzAnswers.values, n, thr=0.15)[0]))


print(
      sum([len(ls)==0 for ls in valid_df.WOzAnswers.values])/len(valid_df.WOzAnswers.values),
      sum([len(ls)==0 for ls in valid_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in valid_df.WOzAnswers.values]) 
      )
print(
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values])/len(test_df.WOzAnswers.values),
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in test_df.WOzAnswers.values]) 
      ) 

### InferSent ###
model_version = 1
MODEL_PATH = "InferSent/encoder/infersent%s.pickle" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'InferSent/dataset/GloVe/glove.840B.300d.txt' if model_version == 1 else 'InferSent/dataset/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

allsentences = []
for item in train_df.Context.values: allsentences.append(item)
for item in train_df.Utterance.values: allsentences.append(item) # used answer too to expand vocabulary
model.build_vocab(allsentences, tokenize=True)

train_embeddings = model.encode(train_df.Utterance.values, bsize=128, tokenize=False, verbose=True)
valid_embeddings = model.encode(valid_df.Context.values, bsize=128, tokenize=False, verbose=True)
test_embeddings = model.encode(test_df.Context.values, bsize=128, tokenize=False, verbose=True)

#Examples mentioned in paper: Q id 20 and 30, answ id 2017 and 77 when the test set was conversations 1 and 2, and I kept empty WoZ answers
np.argsort([cosine(valid_embeddings[1], utt) for utt in train_embeddings], axis=0)[::-1][:5]
np.sort([cosine(valid_embeddings[1], utt) for utt in train_embeddings], axis=0)[::-1][:5]
valid_df.Context[1]
valid_df.WOzAnswers[1]
train_df.Context[99]
train_df.Utterance[99]

### I had to put the WOZ answers given by picking from new data to properly evaluate this. --> think if it makes sense to include the ooc answers too?

# Evaluate InferSent predictor
y = [INFERSENTPredictor_new(valid_embeddings[x], train_embeddings) for x in range(len(valid_embeddings))]
for thr in np.arange(0, 1.0, 0.05):
    print("Recall@10 for thr={}: {:g}".format(thr, evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=10, thr=thr)[0]))

# Test InferSent predictor
y = [INFERSENTPredictor_new(test_embeddings[x], train_embeddings) for x in range(len(test_embeddings))]
for n in [1, 2, 5, 10, 20]:
    print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, test_df.WOzAnswers.values, n, thr=0.7)[0]))
   
    
    
    
######## more turns ###########
train_df, valid_df, test_df = train_test_sets_questions_seqdata_no_ooc(TURNS=1)

model_version = 1
MODEL_PATH = "InferSent/encoder/infersent%s.pickle" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'InferSent/dataset/GloVe/glove.840B.300d.txt' if model_version == 1 else 'InferSent/dataset/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

allsentences = []
for item in train_df.Context.values: allsentences.append(item)
#for item in train_df.Utterance.values: allsentences.append(item) # used answer too to expand vocabulary
model.build_vocab(allsentences, tokenize=True)

train_embeddings = model.encode(train_df.Context.values, bsize=128, tokenize=False, verbose=True)
valid_embeddings = model.encode(valid_df.Context.values, bsize=128, tokenize=False, verbose=True)
test_embeddings = model.encode(test_df.Context.values, bsize=128, tokenize=False, verbose=True)

#Examples mentioned in paper: Q id 20 and 30, answ id 2017 and 77 when the test set was conversations 1 and 2, and I kept empty WoZ answers
np.argsort([cosine(valid_embeddings[1], utt) for utt in train_embeddings], axis=0)[::-1][:5]
np.sort([cosine(valid_embeddings[1], utt) for utt in train_embeddings], axis=0)[::-1][:5]
valid_df.Context[1]
valid_df.WOzAnswers[1]
train_df.Context[145]
train_df.Utterance[178]

### I had to put the WOZ answers given by picking from new data to properly evaluate this. --> think if it makes sense to include the ooc answers too?

# Evaluate InferSent predictor
y = [INFERSENTPredictor_new(valid_embeddings[x], train_embeddings) for x in range(len(valid_embeddings))]
for thr in np.arange(0, 1.0, 0.05):
    print("Recall@10 for thr={}: {:g}".format(thr, evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=10, thr=thr)[2]))

# Test InferSent predictor
y = [INFERSENTPredictor_new(test_embeddings[x], train_embeddings) for x in range(len(test_embeddings))]
for n in [1, 2, 5, 10, 20]:
    print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, test_df.WOzAnswers.values, n, thr=0.85)[0]))
   


