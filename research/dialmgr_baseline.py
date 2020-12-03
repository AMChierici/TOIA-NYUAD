#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:34:58 2020

@author: amc
"""

import pandas as pd
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from collections import defaultdict
from gensim import corpora

knowledgebase = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/MargaritaCorpusKB.csv', encoding='utf-8')

documents = list(pd.Series.unique(knowledgebase.Context))

# remove common words and tokenize
# stoplist = set('what did you how long why were was for a of the and to in'.split())
texts = [
    [word for word in document.lower().split()]# if word not in stoplist]
    for document in documents
]

# remove words that appear only once
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1

# texts = [
#     [token for token in text if frequency[token] > 1]
#     for text in texts
# ]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

from gensim import models
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=20)

doc = "What do you do?"
vec_bow = dictionary.doc2bow(doc.lower().split())n
vec_lsi = lsi[vec_bow]  # convert the query to LSI space
print(vec_lsi)

from gensim import similarities
index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it

index.save('/tmp/margarita.index')
index = similarities.MatrixSimilarity.load('/tmp/margarita.index')

sims = index[vec_lsi]  # perform a similarity query against the corpus
print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples

sims = sorted(enumerate(sims), key=lambda item: -item[1])
for doc_position, doc_score in sims[:20]:
    print(round(doc_score*100, 3), round(match.predict(doc, documents[doc_position])[1]*100, 3), documents[doc_position])




def chat():
  doc = input("Interrogator: ")
  # doc = 'how long have you been in the UAE'
  while doc!="stop":
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print("Margarita: \n", sims[0][1], knowledgebase[knowledgebase['Context'] == documents[sims[0][0]]]['Utterance'].values)
    doc = input("Interrogator: ")

chat()

