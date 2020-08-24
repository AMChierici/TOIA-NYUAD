#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:37:17 2020
#thanks to:
    https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/
    Kendall vs. Spearman: https://datascience.stackexchange.com/questions/64260/pearson-vs-spearman-vs-kendall Kendall more robust, usually lower than Spearman
    
@author: amc
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau


filePath = "/Users/amc/Documents/TOIA-NYUAD/research/data/"

with open(r"{}devTfIdfDialogues.json".format(filePath), "r") as read_file:
    model1 = json.load(read_file)
with open(r"{}devBm25Dialogues.json".format(filePath), "r") as read_file:
    model2 = json.load(read_file)
with open(r"{}devBERTbaseuncasedDialogues.json".format(filePath), "r") as read_file:
    model3 = json.load(read_file)
with open(r"{}devBERTqaRel1to100Dialogues.json".format(filePath), "r") as read_file:
    model4 = json.load(read_file)
with open(r"{}devBERTqaRel1toAllDialogues.json".format(filePath), "r") as read_file:
    model5 = json.load(read_file)
with open(r"{}devGoldDialogues.json".format(filePath), "r") as read_file:
    gold = json.load(read_file)


with open(r"{}mTurkResults_2turns_all.json".format(filePath), 'r') as read_file:
    results = json.load(read_file)

df = pd.DataFrame(gold)

for i in range(len(results)):
    results[i]['isGold'] = 1 if df[df['id']==results[i]['snippet_id']]['model_retrieved_answers'].values[0][0]==results[i]['predicted_answer'] else 0

    
df = pd.DataFrame(model1)
for i in range(len(results)):
    try:
        index = df[df['id']==results[i]['snippet_id']]['model_retrieved_answers'].values[0].index(results[i]['predicted_answer'])
    except ValueError:
        index = -1
    results[i]['model1Score'] = 0 if index==-1 else df[df['id']==results[i]['snippet_id']]['scores'].values[0][index]
    
df = pd.DataFrame(model2)
for i in range(len(results)):
    try:
        index = df[df['id']==results[i]['snippet_id']]['model_retrieved_answers'].values[0].index(results[i]['predicted_answer'])
    except ValueError:
        index = -1
    results[i]['model2Score'] = 0 if index==-1 else df[df['id']==results[i]['snippet_id']]['scores'].values[0][index]
    
df = pd.DataFrame(model3)
for i in range(len(results)):
    try:
        index = df[df['id']==results[i]['snippet_id']]['model_retrieved_answers'].values[0].index(results[i]['predicted_answer'])
    except ValueError:
        index = -1
    results[i]['model3Score'] = 0 if index==-1 else df[df['id']==results[i]['snippet_id']]['scores'].values[0][index]
    
df = pd.DataFrame(model4)
for i in range(len(results)):
    try:
        index = df[df['id']==results[i]['snippet_id']]['model_retrieved_answers'].values[0].index(results[i]['predicted_answer'])
    except ValueError:
        index = -1
    results[i]['model4Score'] = 0 if index==-1 else df[df['id']==results[i]['snippet_id']]['scores'].values[0][index]
    
    
df = pd.DataFrame(model5)
for i in range(len(results)):
    try:
        index = df[df['id']==results[i]['snippet_id']]['model_retrieved_answers'].values[0].index(results[i]['predicted_answer'])
    except ValueError:
        index = -1
    results[i]['model5Score'] = 0 if index==-1 else df[df['id']==results[i]['snippet_id']]['scores'].values[0][index]
    
    
dfResults = pd.DataFrame(results)


for i in dfResults.index:
    dfResults.loc[i, 'worker_ids'] = dfResults.loc[i, 'worker_ids'][0]
    
blackList = []
for workerId in dfResults[(dfResults['isGold']==1) & (dfResults['avg_answer']<=3)]['worker_ids']:
    blackList.append(workerId)

names=[]
for i in range(1,6):
    names.append('model{}Score'.format(i))
    
### CORRELATIONS ###

print("Excluding unqualified workers")
for i in names:
    dfTemp = dfResults[(~dfResults['worker_ids'].isin(blackList)) & (dfResults[i]>0)][['avg_answer', i]]
    print("\n", i, " Correlation: ", dfTemp['avg_answer'].corr(dfTemp[i], 'spearman'))
    
print("\n INCLUDING unqualified workers")
for i in names:
    dfTemp = dfResults[dfResults[i]>0][['avg_answer', i]]
    print("\n", i, " Correlation: ", dfTemp['avg_answer'].corr(dfTemp[i], 'spearman'))

alpha = 0.05  
print("Excluding unqualified workers")
for i in names:
    dfTemp = dfResults[(~dfResults['worker_ids'].isin(blackList)) & (dfResults[i]>0)][['avg_answer', i]]
    coef, p = spearmanr(dfTemp['avg_answer'].values, dfTemp[i].values)
    # interpret the significance
    print("\n", i, " Spearmans Correlation: %.3f" % coef)
    if p > alpha:
    	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
    	print('Samples are correlated (reject H0) p=%.3f' % p)
        
for i in names:
    dfTemp = dfResults[(~dfResults['worker_ids'].isin(blackList)) & (dfResults[i]>0)][['avg_answer', i]]
    coef, p = kendalltau(dfTemp['avg_answer'].values, dfTemp[i].values)
    # interpret the significance
    print("\n", i, " Kendall Correlation: %.3f" % coef)
    if p > alpha:
    	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
    	print('Samples are correlated (reject H0) p=%.3f' % p)
    
    
#dfResults.to_csv('data/dfResults.txt', sep='\t', encoding='utf-8', index=False)
    

### FOR WHICH QUESTIONS THERE IS LESS AGREEABLENESS? CoV by Q, check what they are, then calc. corr for cov high and small ###  

def CoV(x):
    # x is a list or numpy array
    return np.std(x)/np.mean(x)
    
dfResults['lsCovs'] = [CoV(x) for x in dfResults['answers']]
print(dfResults[(~dfResults['worker_ids'].isin(blackList))].describe())

upThr = .514259
loThr = .204124

A = np.unique(dfResults[(~dfResults['worker_ids'].isin(blackList)) & (dfResults['lsCovs']>upThr)]['last_turn'])
B = np.unique(dfResults[(~dfResults['worker_ids'].isin(blackList)) & (dfResults['lsCovs']<=loThr)]['last_turn'])

print(
      len(set(set(A) & set(B))), '\n',
      len(set(A)), '\n',
      len(set(B))
      )
#set(A) - set(set(A) & set(B)) #questions that go well with many answers (check if true)
#set(B) - set(set(A) & set(B)) #question that go well with only a few answers (check if true)
print(
      set(A) - set(set(A) & set(B)), '\n==============\n',
      set(B) - set(set(A) & set(B))
      )
#seems this shows where the model do well or not. Because, the answers raters rated are only the top 10 selections from 5 models, so where there is high disagreement, it means the models produced answers that is hard to agree upon. When there is low agreement, models might do all pretty well. --need to see human ratings by level of agreement, e.g., is it easier to agree on high or low ratings? Or, in other words, do all agree/disagree in correct answers, non correct, or both alike? Moreover, shall we introduce a random top 10 model to study the effect of model selections on disagreemnt?

A = dfResults[(~dfResults['worker_ids'].isin(blackList)) & (dfResults['lsCovs']>upThr)]['predicted_answer']
B = dfResults[(~dfResults['worker_ids'].isin(blackList)) & (dfResults['lsCovs']<=loThr)]['predicted_answer']

print(
      len(set(set(A) & set(B))), '\n',
      len(set(A)), '\n',
      len(set(B))
      )
#set(A) - set(set(A) & set(B)) #answers that go well with many questions (check if true)
#set(B) - set(set(A) & set(B)) #answers that go well with only a few questions (check if true)
print(
      set(A) - set(set(A) & set(B)), '\n==============\n',
      set(B) - set(set(A) & set(B))
      )

#these are groups of answers that generate more disagreement (A) and less disagreements (B).

dfResults.to_csv('data/dfResults.txt', sep='\t', encoding='utf-8', index=False)

###use crowd ratings as annotations and and calc Recall@x
#build dataset like multiturn dialogues with BA1 - 3 (Assuming 3 is still acceptable)

lsCheck = []
for q in set(dfResults['last_turn']):
    #q = 'Oh I see. I see. Yeah. Yeah. So how did you adapt to the Arabic culture?'
    answers = list(dfResults[dfResults['last_turn'] == q]['predicted_answer'])
    scores = list(dfResults[dfResults['last_turn'] == q]['avg_answer'])
    selindices = np.argsort(scores, axis=0)[::-1]
    sortescores = np.sort(scores, axis=0)[::-1]
    mask = selindices[sortescores >= 3.5]
    lsCheck.append(len(mask))
print(max(lsCheck))
    
dicDf = {
    'Q' : [],
    'BA1' : [],
    'BA2' : [],
    'BA3' : [],
    'BA4' : [],
    'BA5' : [],
    'BA6' : [],
    'BA7' : [],
    'BA8' : [],
    'BA9' : [],
    'BA10' : [],
    'BA11' : [],
    'BA12' : [],
    'BA13' : [],
    'BA14' : [],
    'BA15' : []
    }

for q in set(dfResults['last_turn']):
    #q = 'Oh I see. I see. Yeah. Yeah. So how did you adapt to the Arabic culture?'
    dicDf['Q'].append(q)
    answers = list(dfResults[dfResults['last_turn'] == q]['predicted_answer'])
    scores = list(dfResults[dfResults['last_turn'] == q]['avg_answer'])
    selindices = np.argsort(scores, axis=0)[::-1]
    sortescores = np.sort(scores, axis=0)[::-1]
    mask = selindices[sortescores >= 3.5]
    for i in range(15):
        try:
            dicDf['BA{}'.format(1 + i)].append(answers[mask[i]])
        except IndexError:
            dicDf['BA{}'.format(1 + i)].append(np.nan)
            
dfCrowdAnnotations = pd.DataFrame(dicDf)
        
#borrow test_set_questions_ooctrain function from LREC_code_postreview.py and edit index
def transformDialogues(dfCrowdAnnotations, train_df):
    # modified to use index of answers in test WOzAnswers --> replaced with WOzAnswersIDs
    Context, WOzAnswersIDs = [], []
    for example_id in range(len(dfCrowdAnnotations)):
        exampleWOzAnswers = list(dfCrowdAnnotations.iloc[example_id, 1:].values)
#        if not allnull(exampleWOzAnswers):
        tmp_WAs = []
        allAs = list(train_df.Utterance.values)
        for distr in allAs:
            if distr in exampleWOzAnswers:
                tmp_WAs.append(allAs.index(distr))
        Context.append(dfCrowdAnnotations.iloc[example_id, 0])
        WOzAnswersIDs.append(tmp_WAs)
    return pd.DataFrame({'Context':Context, 'WOzAnswers':WOzAnswersIDs})  

# #run LREC_code_postreview until row 300
# valid_df = transformDialogues(dfCrowdAnnotations, train_df)

#run LREC_code_postreview post row 337 to get results
## OCCHIO A come si calcoalno i recall@k. il numero di esempi non dovrebbe essere il numero di uniche q-a pairs annotate? La funzione al momento calcola solo il numero di q...
    ## Edited evaluate_recall_thr to use as no. of examples all the q-a's annotated: Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items). I could also calculate Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k) but need to define threshold (what is recommended vs. what's not)
###Note: cannot really use crowd annotations as dev set annotations because I should change the train set too. I should add more qa pairs in train according to what has been annotated.
#Let's do it!
 
def updateTrain(df):
    #get questions from old train that are answered by margarita's annotations.
    df = df.loc[dfResults.avg_answer >= 3.5, ['last_turn', 'predicted_answer']]
    df.reset_index(level=None, drop=True, inplace=True)
    df.rename(columns = {'last_turn' : 'Context', 'predicted_answer' : 'Utterance'}, inplace = True) 
    return df.loc[:, ['Context', 'Utterance']]

#run LREC_code_postreview until row 277
train_df = updateTrain(dfResults)
valid_df = transformDialogues(dfCrowdAnnotations, train_df)
### using tf-idf we can see that recalls @1,2,5 are actually worse than margarita's annotations, and recalls @10 and @20 are the same. Looks like the technique is robust by change of datasets, by low k recalls are weaker because annotations too generous? Is it worth do the same job for the BERT qa relevance (lots of work)


###use annotations as model and calc Recall@x

# other things to check:
    ###cov for avg answer = good answers, bad, and so-so
    ###correlation for different quantiles of cov
    ###think about measuring the variability of answer per given question (e.g., words look very different / very similar (cosine sim usin bert / use model trained on semantic sim)) --is cov correlated well with this? what I expect is high cov corr with small semantic variability between answers.
    
    









