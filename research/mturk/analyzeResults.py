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
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
import csv
from nltk.metrics import AnnotationTask
import random

filePath = "/Users/amc/Documents/TOIA-NYUAD/research/data/"


def open_json(file_path, file_name):
    with open(r"{}{}".format(file_path, file_name), "r") as read_file:
        return json.load(read_file)


model1 = open_json(filePath, 'testTfIdfDialogues.json')
model2 = open_json(filePath, 'testBm25Dialogues.json')
model3 = open_json(filePath, 'testBERTbaseuncasedDialogues.json')
model4 = open_json(filePath, 'testBERTqaRel1to100Dialogues.json')
model5 = open_json(filePath, 'testBERTqaRel1toAllDialogues.json')
gold = open_json(filePath, 'testGoldDialogues.json')
results = open_json(filePath, 'mTurkResults_2turns_all_testset.json')

# Transform to df
df = pd.DataFrame(gold)

# Add field isGold to results dictionary
for i in range(len(results)):
    lookup = df[df['id']==results[i]['snippet_id']]['model_retrieved_answers']
    pred_ans = results[i]['predicted_answer']
    results[i]['isGold'] = 1 if lookup.values[0][0] == pred_ans else 0


# Add model score to results
# helper function:
def add_score(model, var_name):
    # results is global
    df = pd.DataFrame(model)
    for i in range(len(results)):
        try:
            index = df[df['id'] == results[i]['snippet_id']] \
                ['model_retrieved_answers'].values[0]. \
                index(results[i]['predicted_answer'])
        except ValueError:
            index = -1
        results[i][var_name] = 0 if index==-1 else \
            df[df['id']==results[i]['snippet_id']]['scores'].values[0][index]


add_score(model1, 'model1Score')
add_score(model2, 'model2Score')
add_score(model3, 'model3Score')
add_score(model4, 'model4Score')
add_score(model5, 'model5Score')

workers_blacklist = []
assignments_blacklist = []
thr = 3
for item in results:
    if item['isGold']==1:
        answers = item['answers']
        item['trusted_answers'] = [ans for ans in answers if ans > thr]
        workers_blacklist.extend([item['worker_ids'][answers.index(ans)]
                                  for ans in answers if ans <= thr])
        assignments_blacklist.extend([
item['assignment_ids'][answers.index(ans)]
             for ans in answers if ans <= thr])

workers_blacklist = list(np.unique(workers_blacklist))
assignments_blacklist = list(np.unique(assignments_blacklist))
# Save assignemtn blacklist for rejecting assignments
with open(filePath + "assignments_blacklist_testset.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(assignments_blacklist)

for item in results:
    if item['isGold']==0:
        workers = item['worker_ids']
        answers = item['answers']
        item['trusted_answers'] = [answers[workers.index(worker)] \
                                   for worker in workers if worker \
                                       not in workers_blacklist]

for item in results:
    m = len(item['trusted_answers'])
    if m == len(item['answers']):
        item['all_blacklisted'] = False
        item['trusted_avg_answer'] = item['avg_answer']
    elif m != 0:
        item['all_blacklisted'] = False
        item['trusted_avg_answer'] = \
            sum(item['trusted_answers'])/len(item['trusted_answers'])
    elif m == 0:
        item['all_blacklisted'] = True
        item['trusted_avg_answer'] = None

dfResults = pd.DataFrame(results)

names=[]
for i in range(1,6):
    names.append('model{}Score'.format(i))

### CORRELATIONS ###

print("Excluding unqualified workers (Spearman)")
for i in names:
    dfTemp = dfResults[(~dfResults['all_blacklisted']) & (dfResults[i]>0)][['trusted_avg_answer', i]]
    print("\n", i, " Correlation: ", dfTemp['trusted_avg_answer'].corr(dfTemp[i], 'spearman'))

print("\n INCLUDING unqualified workers (Spearman)")
for i in names:
    dfTemp = dfResults[dfResults[i]>0][['avg_answer', i]]
    print("\n", i, " Correlation: ", dfTemp['avg_answer'].corr(dfTemp[i], 'spearman'))

alpha = 0.05
print("Excluding unqualified workers")
for i in names:
    dfTemp = dfResults[(~dfResults['all_blacklisted']) & (dfResults[i]>0)][['trusted_avg_answer', i]]
    coef, p = spearmanr(dfTemp['trusted_avg_answer'].values, dfTemp[i].values)
    # interpret the significance
    print("\n", i, " Spearmans Correlation: %.3f" % coef)
    if p > alpha:
    	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
    	print('Samples are correlated (reject H0) p=%.3f' % p)

for i in names:
    dfTemp = dfResults[(~dfResults['all_blacklisted']) & (dfResults[i]>0)][['trusted_avg_answer', i]]
    coef, p = kendalltau(dfTemp['trusted_avg_answer'].values, dfTemp[i].values)
    # interpret the significance
    print("\n", i, " Kendall Correlation: %.3f" % coef)
    if p > alpha:
    	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
    	print('Samples are correlated (reject H0) p=%.3f' % p)


dfResults.to_csv('data/dfResultsAll_testset.txt', sep='\t', encoding='utf-8', index=False)

worker_ids = []
for item in results:
    worker_ids.extend(item['worker_ids'])

print('tot workers: {}\n tot ratings: {}'.format(
    len(pd.unique(worker_ids)), len(worker_ids)
    ))

# Print InterAnnotators Agreements
annotations_closest = []
for annotation in results:
    if len(set(annotation['worker_ids']) & set(workers_blacklist)) == 0:
        sorted_labels = sorted(annotation['answers'])
        if abs(sorted_labels[0] - sorted_labels[1]) < abs(sorted_labels[0] - sorted_labels[-1]):
            labels = [sorted_labels[0], sorted_labels[1]]
        else:
            labels = [sorted_labels[-1], sorted_labels[1]]
        random.shuffle(labels)
        for i in range(2):
            coder = 'c' + str(i + 1)
            item = annotation['hit_id']
            label = labels[i]
            annotations_closest.append((coder, item, label))

annotations_lowest = []
for annotation in results:
    if len(set(annotation['worker_ids']) & set(workers_blacklist)) == 0:
        if annotation['worker_ids'] not in workers_blacklist:
            sorted_labels = sorted(annotation['answers'])
            labels = [sorted_labels[0], sorted_labels[1]]
            random.shuffle(labels)
            for i in range(2):
                coder = 'c' + str(i + 1)
                item = annotation['hit_id']
                label = labels[i]
                annotations_lowest.append((coder, item, label))

annotations_highest = []
for annotation in results:
    if len(set(annotation['worker_ids']) & set(workers_blacklist)) == 0:
        sorted_labels = sorted(annotation['answers'])
        labels = [sorted_labels[1], sorted_labels[2]]
        random.shuffle(labels)
        for i in range(2):
            coder = 'c' + str(i + 1)
            item = annotation['hit_id']
            label = labels[i]
            annotations_highest.append((coder, item, label))

annotations_random = []
for annotation in results:
    if len(set(annotation['worker_ids']) & set(workers_blacklist)) == 0:
        sorted_labels = sorted(annotation['answers'])
        labels = random.sample(sorted_labels, 2)
        random.shuffle(labels)
        for i in range(2):
            coder = 'c' + str(i + 1)
            item = annotation['hit_id']
            label = labels[i]
            annotations_random.append((coder, item, label))

print('Weighted Kappa (Cohen, 1968) \n ------------------------- \n \
      Closest two Ratings: {};\n \
      Lowest two Ratings: {};\n \
      Highest two Ratings: {};\n \
      Random two Ratings: {}\n'.format(
        AnnotationTask(data=annotations_closest).weighted_kappa(),
        AnnotationTask(data=annotations_lowest).weighted_kappa(),
        AnnotationTask(data=annotations_highest).weighted_kappa(),
        AnnotationTask(data=annotations_random).weighted_kappa()))

### FOR WHICH QUESTIONS THERE IS LESS AGREEABLENESS? CoV by Q, check what they are, then calc. corr for cov high and small ###

#From here onward, exclude blacklist workers and gold answers. ### DOUBLE CHECK blacklsit WAS CORRECT EARLIER. IT SEEMS NOT ###

dfResults = dfResults[(~dfResults['all_blacklisted']) & (dfResults['isGold'] == 0)]
# And replace answers with trusted answers
dfResults['answers'] = dfResults['trusted_answers']
dfResults = dfResults.drop(columns=['trusted_answers'])

def CoV(x):
    # x is a list or numpy array
    if len(x) > 1:
        res = np.std(x)/np.mean(x)
    else:
        res = np.nan
    return res

dfResults['lsCovs'] = [CoV(x) for x in dfResults['answers']]
print(dfResults.describe())

upThr = .50
loThr = .25

A = np.unique(dfResults[dfResults['lsCovs'] > upThr]['last_turn'])
B = np.unique(dfResults[dfResults['lsCovs'] <= loThr]['last_turn'])

print(
      len(set(set(A) & set(B))), '\n',
      len(set(A)), '\n',
      len(set(B))
      )
# I shall check my metrics only on these subsets!
print(
      # questions that go well with many answers (actually, that generrate more disagreement - either becasue go well with many answers or )
      set(A) - set(set(A) & set(B)), '\n==============\n',
      # questions that go well with only a few answers (rerunning with set2 = this and adding --and annotation['last_turn'] in set2)-- to the if conditions above when computing the interrannotator agrements, we find out that the agreement that improves the most is the Highest two Ratings (it doubles), meaning that the hyp is sound: questions go well with few answers as there is more agreement in the high ratings. set1 have worst agreement (negative!) on the highest 2 ratings.
      set(B) - set(set(A) & set(B))
      )
#seems this shows where the model do well or not. Because, the answers raters rated are only the top 10 selections from 5 models, so where there is high disagreement, it means the models produced answers that is hard to agree upon. When there is low agreement, models might do all pretty well. --need to see human ratings by level of agreement, e.g., is it easier to agree on high or low ratings? Or, in other words, do all agree/disagree in correct answers, non correct, or both alike? Moreover, shall we introduce a random top 10 model to study the effect of model selections on disagreemnt?

A = dfResults[dfResults['lsCovs'] > upThr]['predicted_answer']
B = dfResults[dfResults['lsCovs'] <= loThr]['predicted_answer']

print(
      len(set(set(A) & set(B))), '\n',
      len(set(A)), '\n',
      len(set(B))
      )
print(
      # answers that go well with many questions (check if true -- yes, same experiment as above confirms)
      set(A) - set(set(A) & set(B)), '\n==============\n',
      # answers that go well with only a few questions (check if true)
      set(B) - set(set(A) & set(B))
      )
#these are groups of answers that generate more disagreement (A) and less disagreements (B).

dfResults.to_csv('data/dfResults_qualified_nogold_testset.txt',  \
                 sep='\t', encoding='utf-8', index=False)

###use crowd ratings as annotations and and calc Recall@x
#build dataset like multiturn dialogues with BA1 - etc.
lsCheck = []
# For question in last turn of workers annotations
for q in set(dfResults['last_turn']):
    answers = list(dfResults[dfResults['last_turn'] == q]['predicted_answer'])
    scores = list(dfResults[dfResults['last_turn'] == q]['trusted_avg_answer'])
    selindices = np.argsort(scores, axis=0)[::-1]
    sortescores = np.sort(scores, axis=0)[::-1]
    mask = selindices[sortescores >= 3.5]
    lsCheck.append(len(mask))
nBA = max(lsCheck)

dicDf = {}
dicDf['Q'] = []
for i in range(1, 1+nBA):
    dicDf['BA{}'.format(i)] = []

for q in set(dfResults['last_turn']):
    dicDf['Q'].append(q)
    answers = list(dfResults[dfResults['last_turn'] == q]['predicted_answer'])
    # 1st option
    scores = list(dfResults[dfResults['last_turn'] == q]['trusted_avg_answer'])
    selindices = np.argsort(scores, axis=0)[::-1]
    sortescores = np.sort(scores, axis=0)[::-1]
    mask = selindices[sortescores >= 3.5]
    #2nd option
    #scores = list(dfResults[dfResults['last_turn'] == q]['answers'])
    #mask = [idx for idx, votes in enumerate(scores) if any(votes)>3]
    #3rd option
    #scores = [max(votes) for votes in list(dfResults[dfResults['last_turn'] == q]['answers'])]
    #selindices = np.argsort(scores, axis=0)[::-1]
    #sortescores = np.sort(scores, axis=0)[::-1]
    #mask = selindices[sortescores > 3]
    #4th option
    #scores = [random.choice(votes) for votes in list(dfResults[dfResults['last_turn'] == q]['answers'])]
    #selindices = np.argsort(scores, axis=0)[::-1]
    #sortescores = np.sort(scores, axis=0)[::-1]
    #mask = selindices[sortescores > 3]
    for i in range(nBA):
        try:
            dicDf['BA{}'.format(1 + i)].append(answers[mask[i]])
        except IndexError:
            dicDf['BA{}'.format(1 + i)].append(np.nan)

dfCrowdAnnotations = pd.DataFrame(dicDf)

dfCrowdAnnotations.to_csv('data/dfCrowdAnnotations_opt1_testset.txt', sep='\t', encoding='utf-8', index=False)


# borrow test_set_questions_ooctrain function from LREC_code_postreview.py and edit index
def transformDialogues(dfCrowdAnnotations, train_df):
    # modified to use index of answers in test WOzAnswers --> replaced with WOzAnswersIDs
    Context, WOzAnswersIDs = [], []
    allAs = list(train_df.Utterance.values)
    for example_id in range(len(dfCrowdAnnotations)):
        exampleWOzAnswers = list(dfCrowdAnnotations.iloc[example_id, 1:].values)
#        if not allnull(exampleWOzAnswers):
        tmp_WAs = []
        for distr in allAs:
            if distr in exampleWOzAnswers:
                tmp_WAs.append(allAs.index(distr)) ## INDEX IS BAD: THERE ARE MORE THAN 1 INDEX FOR ONE DISTR
        Context.append(dfCrowdAnnotations.iloc[example_id, 0])
        WOzAnswersIDs.append(tmp_WAs)
    return pd.DataFrame({'Context':Context, 'WOzAnswers':WOzAnswersIDs})


# not sure if train should stay the same or not
# Cannot really use crowd annotations as dev set annotations because I
# should change the train set too. I should add more qa pairs in train
# according to what has been annotated. Let's do it:
# def updateTrain(df):
#     # get questions from old train that are answered by margarita's annotations.
#     df = df.loc[dfResults.trusted_avg_answer >= 3.5, ['last_turn', 'predicted_answer']]
#     df.reset_index(level=None, drop=True, inplace=True)
#     df.rename(columns = {'last_turn' : 'Context', 'predicted_answer' : 'Utterance'}, inplace = True)
#     return df.loc[:, ['Context', 'Utterance']]


# train_df = updateTrain(dfResults)

train_df = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/MargaritaCorpusKB.csv', encoding='utf-8')
valid_df = transformDialogues(dfCrowdAnnotations, train_df)

#train_df.to_csv('data/crowd_train_df_opt1.txt', sep='\t', encoding='utf-8', index=False)

valid_df.to_csv('data/crowd_valid_df_opt1_testset.txt', sep='\t', encoding='utf-8', index=False)


# other things to check:
    ###cov for avg answer = good answers, bad, and so-so
    ###correlation for different quantiles of cov
    ###think about measuring the variability of answer per given question (e.g., words look very different / very similar (cosine sim usin bert / use model trained on semantic sim)) --is cov correlated well with this? what I expect is high cov corr with small semantic variability between answers.











