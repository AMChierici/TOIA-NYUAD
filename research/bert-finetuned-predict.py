#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:41:38 2020

@author: amc
"""

# -*- coding: utf-8 -*-
​
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

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

model_path = '/Users/amc/Documents/fine_tuned_models/bert_text_classification/MRPC/'

class QAmatcher:

    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

    def predict(self, question, answer):
        # Add special tokens takes care of adding [CLS], [SEP] tokens
        inputs = self.tokenizer.encode_plus(question, answer, add_special_tokens=True, return_tensors="pt")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs[0]
        probas = F.softmax(logits[0], dim=0).detach().numpy()
        return probas

    def predict_qapairs(self, questions, answers):
        self.model.eval()
        # predictions = []
        # probas = []
        preds: torch.Tensor = None
        with torch.no_grad():
            # for question, answer in zip(questions, answers):
            inputs = self.tokenizer.batch_encode_plus([(q, a) for q, a in zip(questions, answers)], add_special_tokens=True, return_tensors="pt")
            outputs = self.model(**inputs)
            # outputs = self.model(inputs['input_ids'].to(self.device), token_type_ids=inputs['token_type_ids'].to(self.device))
            logits = outputs[0]
            probas = [F.softmax(logits[k], dim=0).detach().numpy() for k in range(len(logits))]
        return probas

match = QAmatcher(model_path)

##
match.predict("Hello", "Hi")

n=100
pred_probas = match.predict_qapairs(test_df['#1 String'].values[:n], test_df['#2 String'].values[:n])
preds = np.argmax(pred_probas, axis=1)

def random_preds(labels):
    n1 = np.sum(labels)
    n0 = len(labels) - n1
    # n1 = int(len(test_df['Quality'].values)/2)
    # n0 = len(test_df['Quality'].values) - n1
    a = np.array(np.repeat(0, n0))
    b = np.array(np.repeat(1, n1))
    preds = np.concatenate((a,b), axis=None)
    np.random.shuffle(preds)
    return preds

from sklearn.metrics import *
from print_confusion_matrix import *
def print_metrics(y_pred, y_true, proba=True):
    if proba:
        y_pred = [1 if y>.5 else 0 for y in y_pred]
    print("\n Recall: ", recall_score(y_true, y_pred, labels=[1, 0]),
        "\n Balanced Accuracy: ", balanced_accuracy_score(y_true, y_pred),
        "\n Macro Precision: ", precision_score(y_true, y_pred, average='macro', labels=[1, 0]),
        "\n Precision: ", precision_score(y_true, y_pred, labels=[1, 0]),
        "\n F1: ", f1_score(y_true, y_pred, labels=[1, 0]),
        "\n Confusion Matrix:"
        )
    print_confusion_matrix(y_true, y_pred)

preds = random_preds(valid_df['Quality'].values)
print_metrics(preds, valid_df['Quality'].values)

preds = pd.read_csv('/Users/amc/Documents/glue_data/Margarita_1_All_ratio/valid_results_mrpc.txt', sep='\t', encoding='utf-8')['prediction'].values
print_metrics(preds, valid_df['Quality'].values)

error_analysis = valid_df[valid_df['Quality'].values != preds]
error_analysis.to_csv('data/error_analysis.txt', sep='\t', encoding='utf-8', index=True)


import pandas as pd
# change name for train_df, and run LREC code to get the train_df in the same format as the MDC paper
preds = pd.read_csv('/Users/amc/Documents/glue_data/Margarita_1_All_ratio/valid_results_mrpc.txt', sep='\t', encoding='utf-8')['prediction'].values
valid_preds = pd.DataFrame({'q': valid_df['#1 String'].values, 'A': valid_df['#2 String'].values, 'y_pred': preds})


def chat(k=1, list_KB=list_KB):
  query = input("Interrogator: ")
  # query='how long have you been in the UAE'
  while query!="stop":
    predictions = [match.predict(query, A)[1] for A in list_KB]
    np.argsort(predictions, axis=0)[::-1]
    if k == 1:
        KB_index = list(np.argsort(predictions, axis=0)[::-1])[0]
        print("\nAvatar:", list_KB[KB_index])
    else:
        KB_index = list(np.argsort(predictions, axis=0)[::-1])[:k]
        print("\nAvatar:", [list_KB[i] for i in KB_index])
    query = input("Interrogator: ")

chat()



#---
train_corpus = list(np.unique(train_df.Context.values))
from rank_bm25 import BM25Okapi

bm25_step2 = BM25Okapi(tokenized_corpus)
q = valid_df['Q'].values[10]

y = []
for q in valid_df['Q'].values:
    mask = valid_preds['q'] == q
    relevant = valid_preds['y_pred'] == 1
    subsetkb = train_df['Utterance'].isin(valid_preds[mask&relevant]['A'])

    # q-q search
    step1_corpus = train_df[subsetkb].Context.tolist()
    if not step1_corpus:
        step1_corpus = train_corpus
        tokenized_corpus = [doc.split(" ") for doc in step1_corpus]
        bm25_step2 = BM25Okapi(tokenized_corpus)
        step2_rankings = bm25.get_scores(q.split(" "))
        y.append(step2_rankings)
    else:
        tokenized_corpus = [doc.split(" ") for doc in step1_corpus]
        bm25_step2 = BM25Okapi(tokenized_corpus)
        step2_rankings = bm25.get_scores(q.split(" "))

        ids = [list(train_corpus).index(q) for q in step1_corpus]
        y_rankings = []
        for i in range(len(train_corpus)):
            if i in ids:
                y_rankings.append(step2_rankings[ids.index(i)])
            else:
                y_rankings.append(0)
        y.append(y_rankings)


yhat = []
for q in valid_df['Q'].values:
    mask = valid_preds['q'] == q
    relevant = valid_preds['y_pred'] > .5
    subsetkb = train_df['Utterance'].isin(valid_preds[mask&relevant]['A'])

    # q-q search
    step1_corpus = train_df[subsetkb].Context.tolist()
    if not step1_corpus:
        yhat.append([0]*len(train_corpus))
    else:
        ids = [list(train_corpus).index(q) for q in step1_corpus]
        y_rankings = []
        for i, r in zip(range(len(train_corpus)), valid_preds[mask&relevant]['y_pred'].values):
            if i in ids:
                y_rankings.append(r)
            else:
                y_rankings.append(0)
        yhat.append(y_rankings)






