#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 22:13:01 2020

@author: amc
"""


import pandas as pd
import numpy as np

knowledgebase = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/MargaritaCorpusKB.csv', encoding='utf-8')
train_test_dialogues = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/DIALOGUES.csv', encoding='utf-8')

validation_dialogues = train_test_dialogues.loc[train_test_dialogues.Experiment=="TRAIN"]
test_dialogues = train_test_dialogues.loc[train_test_dialogues.Experiment=="TEST"]

KBanswers = list(np.unique(knowledgebase.Utterance.values))
KBquestions = list(np.unique(knowledgebase.Context.values))
validQuestions = list(np.unique(validation_dialogues.Q.values))


import spacy
# Load the installed model "en_core_web_md"
nlp = spacy.load("en_core_web_md")

doc_questions = list(nlp.pipe(KBquestions))
doc_answers = list(nlp.pipe(KBanswers))
doc_validQuestions = list(nlp.pipe(validQuestions))

from spacy.tokens import Doc

annotation_vars = ['BA1', 'BA2', 'BA3', 'BA4', 'BA5', 'BA6']
def get_lowest_annotated_similarity(question):
    tmp = validation_dialogues[validation_dialogues['Q']==question.text][annotation_vars]
    docs = []
    for j in annotation_vars: 
        if not tmp[j].isnull().values[0]:
            docs.append(tmp[j].values[0])
    docs = list(nlp.pipe(docs))
    similarities=[question.similarity(doc) for doc in docs]
    return min(similarities)

Doc.set_extension("lowest_annotated_similarity", getter=get_lowest_annotated_similarity, force=True)

for doc in doc_validQuestions[:3]:
    print(doc.text, '-', doc._.lowest_annotated_similarity)

def get_kb_similarities(question):
    return [question.similarity(kbDoc) for kbDoc in doc_questions]

Doc.set_extension("kb_similarities", getter=get_kb_similarities, force=True)

from itertools import compress
question = doc_validQuestions[36]
def get_distractors(question):
    mask = list(question._.kb_similarities < question._.lowest_annotated_similarity)
    return list(knowledgebase[
        knowledgebase['Context'].isin(
            list(compress(KBquestions, mask))
            )].Utterance.values
        )

Doc.set_extension("distractors", getter=get_distractors, force=True)

####Nice exercise but it works only for questions in the VALID set, not in the KB --because of logic in worw 36////
                

# def get_kb_similarities(question):
#     return [question.similarity(kbDoc) for kbDoc in doc_questions]

# Doc.set_extension("kb_similarities", getter=get_kb_similarities, force=True)
    

# def get_top_6_similar_questions(doc):
#     similarities = [doc.similarity(kbDoc) for kbDoc in doc_questions]
#     rankedIds = np.argsort(similarities, axis=0)[::-1]
#     return [doc_questions[i] for i in rankedIds[:6]]
    
# Doc.set_extension("top_6_similar_questions", getter=get_top_6_similar_questions, force=True)

# for doc in doc_validQuestions[:10]:
#     print(doc.text, '-', doc._.top_6_similar_questions)


# def get_most_similar_questions(doc):
#     similarities = [doc.similarity(kbDoc) for kbDoc in doc_questions]
#     rankedIds = np.argsort(similarities, axis=0)[::-1]
#     return doc_questions[rankedIds[0]].text
    
# Doc.set_extension("most_similar_questions", getter=get_most_similar_qustions, force=True)

# for doc in doc_validQuestions[:10]:
#     print(doc.text, '-', knowledgebase[knowledgebase['Context']==doc._.most_similar_questions].Utterance.values[0])
    
# def chat():    
#   query = nlp(input("Your Query: "))
#   #query=['how long have you been in the UAE']
#   while query.text!="stop":
#     try: 
#         answer = knowledgebase[knowledgebase['Context']==query._.most_similar_questions].Utterance.values[0]
#         print("\n\n======================\n\n")
#         print("Query:", query.text)
#         print("\nAnswer to most similar question in corpus:")
#         print("\n ", answer, " \n===\n")
#     except ValueError:
#         print("some error")
#         break
#     query = nlp(input("Your Query: "))
    
    
# doc = nlp("tell me about your passions")
# [doc.similarity(kbDoc) for kbDoc in doc_questions]
# list(nlp.pipe(KBquestions))
