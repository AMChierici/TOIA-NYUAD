#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:00:27 2020

@author: amc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:35:07 2019

@author: amc
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import collections
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
from itertools import compress
import torch
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
# Load model
nltk.download('stopwords')
ps = SnowballStemmer('english')
warnings.filterwarnings('ignore')
import math 
# !pip install brewer2mpl
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
import sys


zeroturndialogue = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/knowledgebase.csv', encoding='utf-8')
zeroturndialogue = zeroturndialogue.fillna('')
### Clean KB ###
rows, cols = zeroturndialogue.shape

ID, category, context, utterance, tmp = [], [], [], [], []
    
for r in range(rows): 
    rID = zeroturndialogue.loc[r, 'ID']
    rcategory = zeroturndialogue.loc[r, 'Category']
    rquestions = zeroturndialogue.iloc[r, 4:]
    ranswer = zeroturndialogue.loc[r, 'A']
    i = 0
    while i<cols-4 and rquestions[i]!='':
        ID.append(rID)
        category.append(rcategory)
        context.append(rquestions[i].strip())
        utterance.append(ranswer.replace('ALREADY RECORDED //', ''))
        tmp.append(rquestions[i].strip() + " - " + ranswer.replace('ALREADY RECORDED //', ''))
        i += 1

train_df = pd.DataFrame({'ID':ID, 'Category':category, 'Context':context, 'Utterance':utterance, 'tmp':tmp})
#drop rows that contains useless questions
#train_df = train_df[train_df.Context != '*']
train_df = train_df[train_df.Context != 'FILLER']
train_df = train_df[train_df.Context != 'SIRI-PRE']
train_df = train_df[train_df.Context != 'SIRI-POST']
#train_df = train_df[train_df.Category != 'C-ShortAnswers']
#drop duplicates (like "bye - bye bye!", etc.)
train_df.drop_duplicates(subset='tmp', keep='last', inplace=True) 
# and remove tmp
train_df = train_df.drop('tmp', axis=1)
train_df = train_df.reset_index()

#Dor creating resource repo, commment lines 84-86 above (removing filler, siri pre and siri post) and output csv (then removed index and renamed contest, utt as Q, A)
# train_df.to_csv('MargaritaCorpusKB.csv', encoding='utf-8')

#################

## upload dialogues ##
multiturndialogues = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/DIALOGUES.csv', encoding='utf-8')
#update conversation number so that EDU and PER have different count / conversation ID
multiturndialogues.loc[multiturndialogues.Mode=='EDU', 'Conversation'] = multiturndialogues.loc[multiturndialogues.Mode=='EDU', 'Conversation'] + 10
      

#-- functions --#
def test_set_questions_ooctrain(multiturndialogues, train_df):
    # modified to use index of answers in test WOzAnswers --> replaced with WOzAnswersIDs
    Context, WOzAnswersIDs = [], []
    
    for example_id in range(len(multiturndialogues)):
        exampleWOzAnswers = list(multiturndialogues.iloc[example_id, 7:].values)
#        if not allnull(exampleWOzAnswers):
        tmp_WAs = []
        allAs = list(train_df.Utterance.values)
        for distr in allAs:
            if distr in exampleWOzAnswers:
                tmp_WAs.append(allAs.index(distr))
        Context.append(multiturndialogues.iloc[example_id, 4])
        WOzAnswersIDs.append(tmp_WAs)

    
    return pd.DataFrame({'Context':Context, 'WOzAnswers':WOzAnswersIDs})  


def preprocess(text):
            # Stem and remove stopwords
            text = re.sub('[^a-zA-Z]', ' ', text)
            text = text.lower()
            text = text.split()
            text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
            return ' '.join(text)

def allnull(somelist):
    count=0
    for i in somelist:
        if pd.isnull(i):
            count+=1
    return count==len(somelist)

def evaluate_recall_thr(y, y_test, k=10, thr=0.7):
    # modifying this to allow fine-tuning the threshold and counting empty answers
    num_examples = 0
    num_correct_ans = 0
    num_correct_nonans = 0
    for scores, labels in zip(y, y_test):
        predictions = np.argsort(scores, axis=0)[::-1] #added when scores are numbers and not predicted labels
        sorted_scores = np.sort(scores, axis=0)[::-1] #added when scores are numbers and not predicted labels
        above_thr_selections = [item for item, value in zip(predictions[:k], sorted_scores[:k]) if value>thr]
        A, B = set(labels), set(above_thr_selections)
        num_examples += len(A) #num of examples is num of (q, annotated a) pairs. + 1 if no annotated ans exists.
        intersection = A & B
        if len(intersection)>0:
            num_correct_ans += len(list(intersection))
        elif len(intersection)==0 and len(A)==0 and len(B)==0:
            num_correct_nonans += 1
            num_examples += 1
    return (num_correct_ans+num_correct_nonans)/num_examples, num_correct_ans, num_correct_nonans

def evaluate_recall_thr_star(y, y_test, k=10, thr=0.7):
    # modifying this to allow fine-tuning the threshold and counting empty answers
    num_examples = len(y)
    num_correct_ans = 0
    num_correct_nonans = 0
    for scores, labels in zip(y, y_test):
        predictions = np.argsort(scores, axis=0)[::-1] #added when scores are numbers and not predicted labels
        sorted_scores = np.sort(scores, axis=0)[::-1] #added when scores are numbers and not predicted labels
        above_thr_selections = [item for item, value in zip(predictions[:k], sorted_scores[:k]) if value>thr]
#        intersection = set(labels) & set(predictions[:k])
        A, B = set(labels), set(above_thr_selections)
        intersection = A & B
        if len(intersection)>0:
            num_correct_ans += 1 #if there is an intersection, count 1. I.e., at least 1 relevant answer in the top k

        elif len(intersection)==0 and len(A)==0 and len(B)==0:
            num_correct_nonans += 1
    return (num_correct_ans+num_correct_nonans)/num_examples, num_correct_ans, num_correct_nonans

def print_retrieval_metrics(y, valid_df, KBanswers):                         
    query_precisions, query_recalls, ave_precisions, rec_ranks = [], [], [], []
    for query, retrieval_scores in zip(list(valid_df_orig.Q.values), y):
        sorted_retrieval_scores = np.sort(retrieval_scores, axis=0)[::-1]
        columns_annotations = valid_df_orig.columns[valid_df_orig.columns.get_loc('BA1'):]
        relevant_documents = [doc for doc in list(valid_df_orig[columns_annotations].loc[valid_df_orig.Q.values==query].values[0]) if doc==doc]
        if sorted_retrieval_scores[0]==0 or len(relevant_documents)==0:
            sorted_retrieved_documents = []
            relevant_documents = []
            query_recalls.append(0)
            ave_precisions.append(0)
            rec_ranks.append(0)
        else:
            sorted_id_documents = np.argsort(retrieval_scores, axis=0)[::-1]
            sorted_id_retreved_documents = sorted_id_documents[sorted_retrieval_scores > 0]
            sorted_retrieved_documents = [KBanswers[i] for i in sorted_id_retreved_documents]
            columns_annotations = valid_df_orig.columns[valid_df_orig.columns.get_loc('BA1'):]
            relevant_documents = [doc for doc in list(valid_df_orig[columns_annotations].loc[valid_df_orig.Q.values==query].values[0]) if doc==doc]
            query_recalls.append(len(set(relevant_documents) & set(sorted_retrieved_documents)) / len(set(relevant_documents)))    
            p_at_ks, rel_at_ks = [], []
            for k in range(1, 1+len(set(sorted_retrieved_documents))):
                p_at_ks.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:k])) / k)
                rel_at_ks.append(1 if sorted_retrieved_documents[k-1] in relevant_documents else 0)
            ave_precisions.append(sum([p*r for p, r in zip(p_at_ks, rel_at_ks)])/len(set(relevant_documents)))
        if query_recalls[-1]>0:
            for r, doc in enumerate(sorted_retrieved_documents):
                if doc in relevant_documents:
                    rec_ranks.append(1/(1+r))
                    break
            else:
                rec_ranks.append(0)
    print("Mean Reciprocal Rank (MRR): ", np.mean(rec_ranks))
    print("Mean Average Precision (MAP): ", np.mean(ave_precisions))


class TFIDFPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, corpus):
        self.vectorizer.fit(corpus)

    def predict(self, query, documents):
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([query])
        vector_doc = self.vectorizer.transform(documents)
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
#        return np.argsort(result, axis=0)[::-1]
        return result

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
#    return np.linalg.norm(u-v)


def LMPredictor_new(context, utterances):
    # The dot product measures the similarity of the resulting vectors
    result = [cosine(context, utt) for utt in utterances]
    # Sort by top results and return the indices in descending order
#    return np.argsort(result, axis=0)[::-1]
    return result

def bertembed(text):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)    
    ## Note that I shall try different strategies to get a single vector for the entire sentence.
    sentence_embedding = torch.mean(encoded_layers[11], 1)
    #sentence_embedding = torch.median(encoded_layers[11], 1)[0] #REMINDER: 12 layers in base (index 11), 24 layers in large (index 23)
    #sentence_embedding = torch.mode(encoded_layers[11], 1)[0]
    #sentence_embedding = torch.max(encoded_layers[11], 1)[0]
    #sentence_embedding = torch.cat((torch.mean(encoded_layers[11], 1), torch.max(encoded_layers[11], 1)[0]), 1)
    #can add embedding of token [CLS], and of [SEP]
    return(sentence_embedding.tolist())

def saveJsonDialogues(filepath, ga=False):
    valid_df = multiturndialogues.loc[multiturndialogues.Experiment.isin(["TRAIN"])]
    valid_df.reset_index(level=None, drop=True, inplace=True)
    dialogueSet = {
        "id": "",
        "turn0": "",
        "turn1": "",
        "turn2": "",
        "model_retrieved_answers": [],
        "scores": []
        }
    dialogues = []
    for testex in range(1, len(valid_df)):        
        dialogueSet["id"] = testex
        dialogueSet["turn0"] = valid_df.Q[testex-1]
        dialogueSet["turn1"] = valid_df.A[testex-1]
        dialogueSet["turn2"] = valid_df.Q[testex]
        if ga:
            dialogueSet["model_retrieved_answers"] = [valid_df.A[testex]]
            dialogueSet["scores"] = [1]
        else:
            Qids = np.argsort(y[testex], axis=0)[::-1][:10]
            dialogueSet["model_retrieved_answers"] = list(train_df.Utterance[Qids])
            dialogueSet["scores"] = list(np.sort(y[testex], axis=0)[::-1][:10])
        dialogues.append(dialogueSet.copy()) 
    import json
    with open(filepath, 'w') as fout:
        json.dump(dialogues , fout)
        
def outputPred(y, y_test, lsTraincorpus, txtFilepath, intK, lsQuestions):
    lsSortedScores = []
    lsSelections = []
    lsLabels = []
    for lsScores, labels in zip(y, y_test):
        lsTmp = [index for index in set(labels) if index!='']
        lsPredsIndices = list(np.argsort(lsScores, axis=0)[::-1][:intK])
        lsSortedScores.extend(list(np.sort(lsScores, axis=0)[::-1][:intK]))
        lsSelections.extend([lsTraincorpus[i] for i in lsPredsIndices])
        lsTmp = list(set([lsTraincorpus[i] for i in lsTmp]))
        lsTmp.extend(['']*(intK - len(lsTmp)))
        lsLabels.extend(lsTmp)
    lsRepQuestions = []
    for txtQuestion in lsQuestions:
        lsRepQuestions.extend([txtQuestion]*intK)
    df = pd.DataFrame({
        'Question': lsRepQuestions,
        'PredAnswers': lsSelections,
        'Scores': lsSortedScores,
        'AnnotatedAnswers':lsLabels
        })
    df.to_csv(txtFilepath, encoding='utf-8')


#-- EOfn --#  ###NOTE: need to add to the dialogues TRAIN the wizard answers



valid_df_orig = multiturndialogues.loc[multiturndialogues.Experiment.isin(["TRAIN"])]# & multiturndialogues.Mode.isin(["PER"])]
valid_df_orig.reset_index(level=None, drop=True, inplace=True)
valid_df = test_set_questions_ooctrain(valid_df_orig, train_df)

# Train TFIDF predictor
pred = TFIDFPredictor()
train_corpus = train_df.Context.values
pred.train(train_corpus)

y = [pred.predict(valid_df.Context[x], list(train_corpus)) for x in range(len(valid_df))]

# Evaluate TFIDF predictor QUESTION SIMILARITY (use train_df.Utterance.values for ANSWER SIM)
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, valid_df.WOzAnswers.values, n, thr=0)[i]))

saveJsonDialogues(filepath='data/devGoldDialogues.json', ga=True)
saveJsonDialogues('data/devTfIdfDialogues.json')
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr_star(y, valid_df.WOzAnswers.values, n, thr=0)[i]))
  
#valid_df_orig = dfCrowdAnnotations
print_retrieval_metrics(y, valid_df_orig, list(train_df.Utterance.values))
        
###### BM25 ######
# Train BM25 predictor q-q relevance
from rank_bm25 import BM25Okapi

#corpus = list(train_df.Context.values)

tokenized_corpus = [doc.split(" ") for doc in train_corpus]
bm25 = BM25Okapi(tokenized_corpus)

y = [bm25.get_scores(valid_df.Context[x].split(" ")) for x in range(len(valid_df))]

# Evaluate BM25 predictor
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, valid_df.WOzAnswers.values, n, thr=0)[i]))

saveJsonDialogues('data/devBm25Dialogues.json')

for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr_star(y, valid_df.WOzAnswers.values, n, thr=0)[i]))  
        
#valid_df_orig = dfCrowdAnnotations
print_retrieval_metrics(y, valid_df_orig, list(train_df.Utterance.values))


########## USING BERT ##########
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

#test = bertembed("hi, how are you?")

# QUESTION SIMILARITY
train_embeddings = []
for text in train_corpus:
    train_embeddings.append(bertembed(text)[0])
train_embeddings = np.array(train_embeddings)
valid_embeddings = []
for text in valid_df.Context.values:
    valid_embeddings.append(bertembed(text)[0])
valid_embeddings = np.array(valid_embeddings)

 
y = [LMPredictor_new(valid_embeddings[x], train_embeddings) for x in range(len(valid_embeddings))]
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, valid_df.WOzAnswers.values, n, thr=0)[i]))
#MEDIAN pooling better than mean and close to TFIDF on test set --need more error analysis because HP: BERT does better selection than tfidf even if wrong question (second best predition by bert better than second best by tfidf). Mode pretty shit. Max better than mode, worse than mean. mean-max not great too. Bert cased doesn't make it better. speach in practice will be converted to lowercase text anyway. Large BERT doesn't bring improvements

saveJsonDialogues('data/devBERTbaseuncasedDialogues.json')
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr_star(y, valid_df.WOzAnswers.values, n, thr=0)[i]))

#valid_df_orig = dfCrowdAnnotations
print_retrieval_metrics(y, valid_df_orig, list(train_df.Utterance.values))



# qarelevance
preds = pd.read_csv('/Users/amc/Documents/glue_data/Margarita_1_100_ratio/valid_results_mrpc_proba.txt', sep='\t', encoding='utf-8')['prediction'].values
###DO run toia_data_processor.py until row 166
valid_preds = pd.DataFrame({'q': valid_df['#1 String'].values, 'A': valid_df['#2 String'].values, 'y_pred': preds})
###DO re-run this script until row 312 or analyzeResults.py row 268 for view from the crowd //rewrite this. I need to put back the old train_df and the old valid_df. Need to simply to save them under a different name before running toia_data_prcessor.py and then put back 
y = []
for i in range(len(valid_df_orig)):
    ranks=[]
    for j in train_corpus:
        answers = train_df[train_df['Context']==j]['Utterance'].values[0]
        rank = valid_preds[(valid_preds['A']==answers) & (valid_preds['q']==valid_df_orig.loc[i, 'Q'])]['y_pred'].values[0]      #change 'Q' to 'Context' when running for Crowd annotations
        ranks.append(rank)
    y.append(ranks)
    
valid_df = test_set_questions_ooctrain(valid_df_orig, train_df)

for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, valid_df.WOzAnswers.values, n, thr=0)[i]))

saveJsonDialogues('data/devBERTqaRel1to100Dialogues.json')

for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr_star(y, valid_df.WOzAnswers.values, n, thr=0)[i]))

outputPred(y, valid_df.WOzAnswers.values, train_df.Utterance.values, '/Users/amc/Documents/TOIA-NYUAD/research/data/devBERTqaRel1to100Results.csv', 20, valid_df.Context.values)
#note that if instead of train_df.Utterance.values we use train_corpus or train_df.Context.values (which is the same thing), we get the questions instead of the answers

valid_df_orig = dfCrowdAnnotations
print_retrieval_metrics(y, valid_df_orig, list(train_df.Utterance.values))


        
#RANDOM
import random as rd
rd.seed(2020)
tmp = list(range(1,1+len(train_corpus)))
y_random = [rd.sample(tmp, len(tmp)) for x in range(len(valid_df))] 
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y_random, valid_df.WOzAnswers.values, n, thr=0)[i]))
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr_star(y_random, valid_df.WOzAnswers.values, n, thr=0)[i]))

#valid_df_orig = dfCrowdAnnotations
print_retrieval_metrics(y_random, valid_df_orig, list(train_df.Utterance.values))



###use annotations as model and calc Recall@x
# valid_df, dfResults from analyzeResults.py
# train_df from this script

# this is human performance
valid_df_orig = dfCrowdAnnotations
lsTrainAnswers = list(train_df.Utterance.values)
y_crowd = []
for q in list(valid_df.Context.values):
    answers = list(dfResults[dfResults['last_turn'] == q]['predicted_answer'])
    train_idxs = [lsTrainAnswers.index(ans) for ans in answers]
    #1st option
    scores = list(dfResults[dfResults['last_turn'] == q]['avg_answer'])
    lsTmp = [0]*len(lsTrainAnswers)
    for i, v in enumerate(train_idxs):
        lsTmp[v] = scores[i]
    y_crowd.append(lsTmp)
 
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y_crowd, valid_df.WOzAnswers.values, n, thr=0)[i]))
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr_star(y_crowd, valid_df.WOzAnswers.values, n, thr=0)[i]))

print_retrieval_metrics(y_crowd, valid_df_orig, list(train_df.Utterance.values))

# go back to valid_df with Margarita's annotations.    
# run code here up to valid_df and valid_df_orig definition and subset on questions we have
valid_df_orig = valid_df_orig[valid_df_orig['Q'].isin(dfCrowdAnnotations.Q.values)]
valid_df = valid_df[valid_df['Context'].isin(dfCrowdAnnotations.Q.values)]
y_crowd = []
for q in list(valid_df.Context.values):
    answers = list(dfResults[dfResults['last_turn'] == q]['predicted_answer'])
    train_idxs = [lsTrainAnswers.index(ans) for ans in answers]
    #1st option
    scores = list(dfResults[dfResults['last_turn'] == q]['avg_answer'])
    lsTmp = [0]*len(lsTrainAnswers)
    for i, v in enumerate(train_idxs):
        lsTmp[v] = scores[i]
    y_crowd.append(lsTmp)
 
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y_crowd, valid_df.WOzAnswers.values, n, thr=0)[i]))
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr_star(y_crowd, valid_df.WOzAnswers.values, n, thr=0)[i]))

print_retrieval_metrics(y_crowd, valid_df_orig, list(train_df.Utterance.values))


#### for debugging #####    


query_recalls, ave_precisions, rec_ranks = [], [], []
# query = "Hello."
# retrieval_scores = y[0]
KBanswers = list(train_df.Utterance.values)

for query, retrieval_scores in zip(list(valid_df_orig.Q.values), y):
    sorted_retrieval_scores = np.sort(retrieval_scores, axis=0)[::-1]
    columns_annotations = valid_df_orig.columns[valid_df_orig.columns.get_loc('BA1'):]
    relevant_documents = [doc for doc in list(valid_df_orig[columns_annotations].loc[valid_df_orig.Q.values==query].values[0]) if doc==doc]
    if sorted_retrieval_scores[0]==0 or len(relevant_documents)==0:
        sorted_retrieved_documents = []
        relevant_documents = []
        query_recalls.append(0)
        ave_precisions.append(0)
        rec_ranks.append(0)
    else:
        sorted_id_documents = np.argsort(retrieval_scores, axis=0)[::-1]
        sorted_id_retreved_documents = sorted_id_documents[sorted_retrieval_scores > 0]
        sorted_retrieved_documents = [KBanswers[i] for i in sorted_id_retreved_documents]
        query_recalls.append(len(set(relevant_documents) & set(sorted_retrieved_documents)) / len(set(relevant_documents)))    
        p_at_ks, rel_at_ks = [], []
        for k in range(1, 1+len(set(sorted_retrieved_documents))):
            p_at_ks.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:k])) / k)
            rel_at_ks.append(1 if sorted_retrieved_documents[k-1] in relevant_documents else 0)
        ave_precisions.append(sum([p*r for p, r in zip(p_at_ks, rel_at_ks)])/len(set(relevant_documents)))
    if query_recalls[-1]>0:
        for r, doc in enumerate(sorted_retrieved_documents):
            if doc in relevant_documents:
                rec_ranks.append(1/(1+r)) 
                break # breaking when first match found --note this uses the First RR, whereas in our case it may be more appropriate to use the TOTAL RR as defined in http://www.lrec-conf.org/proceedings/lrec2002/pdf/301.pdf
        else:
            rec_ranks.append(0)
print("Mean Average Precision (MAP): ", np.mean(ave_precisions))
print("Mean Reciprocal Rank (MRR): ", np.mean(rec_ranks))