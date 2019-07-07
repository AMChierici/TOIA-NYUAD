#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:51:17 2019
Stats on data sets
@author: amc
"""

import numpy as np
import pandas as pd
import create_datasets
from create_datasets import *

train_df = zeroturndialogue.rename(columns = {"Question":"Context", "Answer": "Utterance"})

print('# utterances = {}'.format(len(train_df.Utterance.values)))
print('# words = {}'.format(sum([len(tokens) for tokens in [utt.split() for utt in train_df.Utterance.values]])
      + sum([len(tokens) for tokens in [utt.split() for utt in train_df.Context.values]])))
print('avg. words per utterance = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in train_df.Utterance.values]])))
print('avg. words per context = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in train_df.Context.values]])))
      


print('# utterances = {}'.format(len(multiturndialogues.A.values)))
print('# words = {}'.format(sum([len(tokens) for tokens in [utt.split() for utt in multiturndialogues.A.values]])
      + sum([len(tokens) for tokens in [utt.split() for utt in multiturndialogues.Q.values]])))
print('avg. words per utterance = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in multiturndialogues.A.values]])))
print('avg. words per context = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in multiturndialogues.Q.values]])))
           

conversation_IDs = np.unique(multiturndialogues.ConversationID)

print('min # turns per dialogue = {}'.format(min([len(multiturndialogues.loc[multiturndialogues.ConversationID==i]) for i in conversation_IDs])))
      
print('avg # turns per dialogue = {}'.format(np.mean([len(multiturndialogues.loc[multiturndialogues.ConversationID==i]) for i in conversation_IDs])))

      
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

print('# utterances = {}'.format(len(valid_df.Context.values)))
print('# utterances = {}'.format(len(test_df.Context.values)))
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
      
train_df, valid_df, test_df = train_test_sets_questions_seqdata_no_ooc(TURNS=0)
print('# utterances = {}'.format(len(train_df.Context.values)))
print('# utterances = {}'.format(len(valid_df.Context.values)))
print('# utterances = {}'.format(len(test_df.Context.values)))
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


      
train_df, valid_df, test_df = train_test_sets_questions_seqdata_no_ooc(TURNS=1)  
print('# utterances = {}'.format(len(train_df.Context.values)))
print('# utterances = {}'.format(len(valid_df.Context.values)))
print('# utterances = {}'.format(len(test_df.Context.values)))
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
