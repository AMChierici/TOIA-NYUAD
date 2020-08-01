#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:37:17 2020

@author: amc
"""

import json
import matplotlib.pyplot as plt
import pandas as pd

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


with open(r"{}mTurkResults_2turns_first100.json".format(filePath), 'r') as read_file:
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

print("Excluding unqualified workers")
for i in names:
    dfTemp = dfResults[(~dfResults['worker_ids'].isin(blackList)) & (dfResults[i]>0)][['avg_answer', i]]
    print("\n", i, " Correlation: ", dfTemp['avg_answer'].corr(dfTemp[i]))
    
print("\n INCLUDING unqualified workers")
for i in names:
    dfTemp = dfResults[dfResults[i]>0][['avg_answer', i]]
    print("\n", i, " Correlation: ", dfTemp['avg_answer'].corr(dfTemp[i]))
    
dfResults.to_csv('data/dfResults.txt', sep='\t', encoding='utf-8', index=False)
    

    