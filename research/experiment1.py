#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  26 19:06:40 2019

Create dataset to feed into a CNN

[1] Shen, Y., He, X., Gao, J., Deng, L., and Mesnil, G. 2014. A latent semantic model
#         with convolutional-pooling structure for information retrieval. In CIKM, pp. 101-110.
#         http://research.microsoft.com/pubs/226585/cikm2014_cdssm_final.pdf

Tot tell to Nizar: I want to apply the model in [1] treating it as purely IR. Then I apply Sequence model to treat it as sequential dialogue. Then I run a UX test and see which dialogue mgr performs best for the user. Both datasets are original as Margarita ranks the sequential as well as the IR so those are the labels. BTW, if we want to reproduce the ranking, does the model need to be IR? probably yes, but the question is - does the sequence give me better ranking for the new dataset?

One more thing: anyway the IR for the 'out of context' (old) dataset must be made state of the art so we can get users to use it, click on selected perfectly (5) to bad (0) answer and we record the user labelling over time. But, when the bad selection is because the db does not have an answer to that qs? This comes from Margarita labelling - add labele "there is no answer".


@author: amc
"""

 # -------------------- script for A.I. -----------------------#
import numpy as np
import pandas as pd
import re

def preprocess(text):
            # Remove punctuation and take lower case.
            # text = "I saw the boy coming over."
            text = re.sub('[^a-zA-Z]', ' ', text)
            text = text.lower()
            text = re.sub(' ', '# #', ' '+text+' ')
            return text.split()

        
dataset1 = pd.read_csv('interview1.csv', encoding='ISO-8859-1')
dataset2 = pd.read_csv('interview2.csv', encoding='ISO-8859-1')
dataset3 = pd.read_csv('interview3.csv', encoding='ISO-8859-1')

dataset = pd.concat([dataset1, dataset2, dataset3])
#Reset index otherwise during the loop below we select multiple rows (pandas.concat results in repeated indices)
dataset = dataset.reset_index(drop=True)

vocabulary = []
for i in range(0, len(dataset)):
    answer = preprocess(dataset['A'][i])
    answer = np.array(answer)
    answer = answer[answer!='#']
    answer = answer[answer!='##']    
    #### get the moving window and save voc. then unique and count unique values
    vocabulary.append(answer)      

# Build steps for word2vector


def word2vector(word):
    """
    Argument:
    word -- a numpy array of shape (..., ..., ...)
    
    Returns:
    v -- a vector of shape (...*...*..., 1)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    v = word.reshape((word.shape[0]*word.shape[1]*word.shape[2], 1))
    ### END CODE HERE ###
    
    return v


