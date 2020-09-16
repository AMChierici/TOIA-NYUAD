#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 22:13:01 2020

@author: amc
"""

import pandas as pd
import numpy as np
# import spacy
import spacy_sentence_bert
from spacy.tokens import Doc

kb_file = '/Users/amc/Documents/TOIA-NYUAD/research/MargaritaCorpusKB.csv'
dialogues_file = '/Users/amc/Documents/TOIA-NYUAD/research/DIALOGUES.csv'

knowledgebase = pd.read_csv(kb_file, encoding='utf-8')
train_test_dialogues = pd.read_csv(dialogues_file, encoding='utf-8')

validation_dialogues = train_test_dialogues.loc[
    train_test_dialogues.Experiment == "TRAIN"]
test_dialogues = train_test_dialogues.loc[
    train_test_dialogues.Experiment == "TEST"]

KBanswers = list(np.unique(knowledgebase.Utterance.values))
KBquestions = list(np.unique(knowledgebase.Context.values))
validQuestions = list(np.unique(validation_dialogues.Q.values))

# OPTION1 Load the model en_core_web_md or en_trf_robertabase_lg (remember
# to install first!). For OPT1 uncomment import spacy (row 12)
# nlp = spacy.load("en_trf_bertbaseuncased_lg")  # en_trf_robertabase_lg

# OPTION2 Use Sentence BERT Embedding
nlp = spacy_sentence_bert.load_model('en_bert_base_nli_mean_tokens')

doc_questions = list(nlp.pipe(KBquestions))
doc_answers = list(nlp.pipe(KBanswers))
doc_validQuestions = list(nlp.pipe(validQuestions))


def get_kb_similarities(question):
    return [question.similarity(kbDoc) for kbDoc in doc_questions]


Doc.set_extension("kb_similarities", getter=get_kb_similarities, force=True)


def get_top_6_similar_questions(doc):
    similarities = [doc.similarity(kbDoc) for kbDoc in doc_questions]
    rankedIds = np.argsort(similarities, axis=0)[::-1]
    return [doc_questions[i] for i in rankedIds[:6]]


Doc.set_extension("top_6_similar_questions",
                  getter=get_top_6_similar_questions, force=True)

for doc in doc_validQuestions[:10]:
    print('\n', doc.text, '-', doc._.top_6_similar_questions)


def get_most_similar_questions(doc):
    similarities = [doc.similarity(kbDoc) for kbDoc in doc_questions]
    rankedIds = np.argsort(similarities, axis=0)[::-1]
    return doc_questions[rankedIds[0]].text


Doc.set_extension("most_similar_questions", getter=get_most_similar_questions,
                  force=True)


for doc in doc_validQuestions[:10]:
    print('\n', doc.text, '-',
          knowledgebase[knowledgebase['Context'] == doc._.
                        most_similar_questions].Utterance.values[0])


def chat():
    query = nlp(input("Your Query: "))
    # query=['how long have you been in the UAE']
    while query.text != "stop":
        try:
            answer = knowledgebase[
                knowledgebase['Context'] ==
                query._.most_similar_questions].Utterance.values[0]
            print("\n\n======================\n\n")
            print("Query:", query.text)
            print("\nAnswer to most similar question in corpus:")
            print("\n ", answer, " \n===\n")
        except ValueError:
            print("some error")
            break
        query = nlp(input("Your Query: "))
