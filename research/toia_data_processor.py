#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:26:12 2020

@author: amc

Purpose: Organize the TOIA data sets in the same format as the MRPC GLUE task.
The download link for the MRPC data sets are taken from: https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e.

* dev_ids.tsv: https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc'
* msr_paraphrase_train.txt: https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt
* msr_paraphrase_test.txt: https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt
"""
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

import pandas as pd
import numpy as np
from eda import *

def allnull(somelist):
    count=0
    for i in somelist:
        if pd.isnull(i):
            count+=1
    return count==len(somelist)

knowledgebase = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/MargaritaCorpusKB.csv', encoding='utf-8')
train_test_dialogues = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/DIALOGUES.csv', encoding='utf-8')

validation_dialogues = train_test_dialogues.loc[train_test_dialogues.Experiment=="TRAIN"]
test_dialogues = train_test_dialogues.loc[train_test_dialogues.Experiment=="TEST"]

KBanswers = list(np.unique(knowledgebase.Utterance.values))

# unComment for 1to100
# no_distr_to_sample = 100
#
Context, Utterance, Labels= [], [], []
for example_id in range(len(knowledgebase)):
    Context.append(knowledgebase.Context.values[example_id])
    Utterance.append(knowledgebase.Utterance.values[example_id])
    Labels.append(1)
    id_to_exclude = KBanswers.index(knowledgebase.Utterance.values[example_id])
    tmp_distractors = [KBanswers[i] for i in
            np.array(range(len(KBanswers)))
            [np.isin(range(len(KBanswers)), id_to_exclude, invert=True)]
            ]
    # unComment for 1to100
    # np.random.seed(example_id)
    #
    for answer in tmp_distractors:  # Use for 1to100: for answer in np.random.choice(tmp_distractors, no_distr_to_sample, replace=False):
                Context.append(knowledgebase.Context.values[example_id])
                Utterance.append(str(answer))
                Labels.append(0)

train_df = pd.DataFrame({'#1 String':Context, '#2 String':Utterance, 'Quality':Labels})

# comment for 1to100
train_df.to_csv('data/tmp.txt', sep='\t', encoding='utf-8', index=False, header=False)
#

# unComment for 1to100. for 1toAll Use augment.py instead
# for i in range(len(train_df)):
#     try:
#         tmp_df = pd.DataFrame({
#             '#1 String': eda(train_df['#1 String'].values[i], .2, .2, .2, .2, 15),
#             '#2 String': list(np.repeat(train_df['#2 String'].values[i], 16)),
#             'Quality': list(np.repeat(train_df['Quality'].values[i], 16))
#             })
#         train_df = train_df.append(tmp_df)
#     except Exception:
#         pass
#

# comment for 1to100
train_df = pd.read_csv('data/eda_tmp.txt', sep='\t', encoding='utf-8', header=None)
# add back title
train_df.columns = ['#1 String', '#2 String', 'Quality']
#

# Shuffle
train_df = train_df.sample(frac=1, random_state=1234).reset_index(drop=True)



# validation_dialogues.columns.get_loc('BA1')
# validation_dialogues.columns.get_loc('Q')

Context, WOzAnswers, Labels= [], [], []
for example_id in range(len(validation_dialogues)):
    exampleWOzAnswers = list(validation_dialogues.iloc[example_id, 7:].values)
    ids_to_exclude = []
    if allnull(exampleWOzAnswers):
        for a in KBanswers:
            Context.append(validation_dialogues.iloc[example_id, 4])
            WOzAnswers.append(a)
            Labels.append(0)
    else:
        for answer in exampleWOzAnswers:
            if not pd.isnull(answer):
                Context.append(validation_dialogues.iloc[example_id, 4])
                WOzAnswers.append(answer)
                Labels.append(1)
                ids_to_exclude.append(KBanswers.index(answer))
        #
        tmp_distractors = [KBanswers[i] for i in
                np.array(range(len(KBanswers)))
                [np.isin(range(len(KBanswers)), ids_to_exclude, invert=True)]
                ]
        #
        for a in tmp_distractors:
            Context.append(validation_dialogues.iloc[example_id, 4])
            WOzAnswers.append(a)
            Labels.append(0)

valid_df = pd.DataFrame({'#1 String':Context, '#2 String':WOzAnswers, 'Quality':Labels})


Context, WOzAnswers, Labels= [], [], []
for example_id in range(len(test_dialogues)):
    exampleWOzAnswers = list(test_dialogues.iloc[example_id, 7:].values)
    ids_to_exclude = []
    if allnull(exampleWOzAnswers):
        for a in KBanswers:
            Context.append(test_dialogues.iloc[example_id, 4])
            WOzAnswers.append(a)
            Labels.append(0)
    else:
        for answer in exampleWOzAnswers:
            if not pd.isnull(answer):
                Context.append(test_dialogues.iloc[example_id, 4])
                WOzAnswers.append(answer)
                Labels.append(1)
                ids_to_exclude.append(KBanswers.index(answer))
        #
        tmp_distractors = [KBanswers[i] for i in
                np.array(range(len(KBanswers)))
                [np.isin(range(len(KBanswers)), ids_to_exclude, invert=True)]
                ]
        #
        for a in tmp_distractors:
            Context.append(test_dialogues.iloc[example_id, 4])
            WOzAnswers.append(a)
            Labels.append(0)

test_df = pd.DataFrame({'#1 String':Context, '#2 String':WOzAnswers, 'Quality':Labels})

# Create indices and output data sets
train_df['#1 ID'] = 100 + train_df.index.values
train_df['#2 ID'] = max(train_df['#1 ID']) + 2 + train_df.index.values
valid_df['#1 ID'] = max(train_df['#2 ID']) + 2 + valid_df.index.values
valid_df['#2 ID'] = max(valid_df['#1 ID']) + 2 + valid_df.index.values
valid_df.to_csv('data/valid_dev2valid_preds.tsv', sep='\t', encoding='utf-8', index=False)
test_df['#1 ID'] = max(valid_df['#2 ID']) + 2 + test_df.index.values
test_df['#2 ID'] = max(test_df['#1 ID']) + 2 + test_df.index.values
test_df.to_csv('data/test_dev2test_preds.tsv', sep='\t', encoding='utf-8', index=False)

# Use train train, dev as dev, and predict on test data.
valid_df[["#1 ID", "#2 ID"]].to_csv('data/dev_ids.tsv', sep='\t', encoding='utf-8', index=False, header=False)
out_train_data = train_df.append(valid_df)
out_train_data[["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"]].to_csv('data/msr_paraphrase_train.txt', sep='\t', encoding='utf-8', index=False)
test_df[["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"]].to_csv('data/msr_paraphrase_test.txt', sep='\t', encoding='utf-8', index=False)


np.sum(test_df['Quality'])
416/137490

###################
#FIX SPANISH ENCODING --va a capo prima di ventido anos.

###-clear space-###
del(train_df, ar_msk)
###################