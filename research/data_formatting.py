#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:19:59 2019

@author: amc
"""
    
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
import collections

ps = SnowballStemmer('english')

def preprocess(text):
            # Stem and remove stopwords
            text = re.sub('[^a-zA-Z]', ' ', text)
            text = text.lower()
            text = text.split()
            text = [ps.stem(word) for word in text]# if not word in set(stopwords.words('english'))]
            return ' '.join(text)


oneturndialogue = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/Wizard-of-Oz-dataset - Test Questions.csv', encoding='utf-8-sig')
dialogue1 = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/interview1.csv', encoding='ISO-8859-1')
dialogue2 = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/interview2.csv', encoding='ISO-8859-1')
dialogue3 = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/interview3.csv', encoding='ISO-8859-1')
dialogue4p = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/interview4p.csv', encoding='ISO-8859-1')

for i in range(len(dialogue1)):
    dialogue1.iloc[i, 0] = preprocess(dialogue1.iloc[i, 0])
    dialogue1.iloc[i, 1] = preprocess(dialogue1.iloc[i, 1])
    
for i in range(len(dialogue2)):
    dialogue2.iloc[i, 0] = preprocess(dialogue2.iloc[i, 0])
    dialogue2.iloc[i, 1] = preprocess(dialogue2.iloc[i, 1])
    
for i in range(len(dialogue3)):
    dialogue3.iloc[i, 0] = preprocess(dialogue3.iloc[i, 0])
    dialogue3.iloc[i, 1] = preprocess(dialogue3.iloc[i, 1])

for i in range(len(dialogue4p)):
    dialogue4p.iloc[i, 0] = preprocess(dialogue4p.iloc[i, 0])
    dialogue4p.iloc[i, 1] = preprocess(dialogue4p.iloc[i, 1])

#sample context windows - a window is a number of turns, defined as 1qs and 1ans
#actually, make the window from full dialogue to min 3plets: 1qs, 1a, 1qs - thene we can try random window size, or just one size sliding, e.g. 3, 4, 5, ...etc.
# To start with, I pick 2 2plets: Q-A-Q-A-Q-A-Q +A
TURNS = 2   
tmp = dialogue1['Q'] + ' <eos> ' + dialogue1['A']
Q = []
A = []
for i in range(len(dialogue1)-TURNS):
    Q.append(' <eos> '.join(tmp.loc[i:(i+(TURNS-2))]))
    Q[-1] += ' <eos> ' + tmp.loc[i+(TURNS-1)].split(' <eos> ')[0]
    A.append(tmp.loc[i+(TURNS-1)].split(' <eos> ')[1])
    
df1 = pd.DataFrame({'Context':Q, 'Utterance':A})

#repeat with dialogue2
tmp = dialogue2['Q'] + ' <eos> ' + dialogue2['A']
Q = []
A = []
for i in range(len(dialogue2)-TURNS):
    Q.append(' <eos> '.join(tmp.loc[i:(i+(TURNS-2))]))
    Q[-1] += ' <eos> ' + tmp.loc[i+(TURNS-1)].split(' <eos> ')[0]
    A.append(tmp.loc[i+(TURNS-1)].split(' <eos> ')[1])

df2 = pd.DataFrame({'Context':Q, 'Utterance':A})

train_df = df1.append(df2, ignore_index = True) 

#repeat with dialogue4p
tmp = dialogue4p['Q'] + ' <eos> ' + dialogue4p['A']
Q = []
A = []
for i in range(len(dialogue4p)-TURNS):
    Q.append(' <eos> '.join(tmp.loc[i:(i+(TURNS-2))]))
    Q[-1] += ' <eos> ' + tmp.loc[i+(TURNS-1)].split(' <eos> ')[0]
    A.append(tmp.loc[i+(TURNS-1)].split(' <eos> ')[1])
df3 = pd.DataFrame({'Context':Q, 'Utterance':A})

train_df = train_df.append(df3, ignore_index = True) 

#repeat with dialogue3 for building test df
tmp = dialogue3['Q'] + ' <eos> ' + dialogue3['A']
Q = []
A = []
for i in range(len(dialogue3)-TURNS):
    Q.append(' <eos> '.join(tmp.loc[i:(i+(TURNS-2))]))
    Q[-1] += ' <eos> ' + tmp.loc[i+(TURNS-1)].split(' <eos> ')[0]
    A.append(tmp.loc[i+(TURNS-1)].split(' <eos> ')[1])
N = len(A)
test_df = pd.DataFrame({'Context':Q, 'Ground Truth Utterance':A})
    
# Define no. of distractors
#D = 9
D = len(oneturndialogue)
for i in range(D):
    rows = np.random.choice(oneturndialogue.index.values, N)
    test_df['Distractor_'+str(i)] = oneturndialogue.loc[rows, 'A'].values
    


np.savetxt('./dialogue_data/from.txt', train_df['Context'].tolist(), delimiter=',', fmt='%5s')
np.savetxt('./dialogue_data/to.txt', train_df['Utterance'].tolist(), delimiter=',', fmt='%5s')


# Data Preparation

def build_dataset(words, n_words):
    count = [['<unk>', 0], ['<pad>', 1], ['<eos>', 2]]#, ['<go>', 3]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


file_path = './dialogue_data/'

with open(file_path+'from.txt', 'r') as fopen:
    text_from = fopen.read().lower().split('\n')
with open(file_path+'to.txt', 'r') as fopen:
    text_to = fopen.read().lower().split('\n')
print('len from: %d, len to: %d'%(len(text_from), len(text_to)))


concat_from = ' '.join(text_from).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, human_vocab, inv_human_vocab = build_dataset(concat_from, vocabulary_size_from)
print('vocab from size: %d'%(vocabulary_size_from))
print('Most common words', count_from[4:10])
print('Sample data', data_from[:10], [inv_human_vocab[i] for i in data_from[:10]])



concat_to = ' '.join(text_to).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, machine_vocab, inv_machine_vocab = build_dataset(concat_to, vocabulary_size_to)
print('vocab to size: %d'%(vocabulary_size_to))
print('Most common words', count_to[4:10])
print('Sample data', data_to[:10], [inv_machine_vocab[i] for i in data_to[:10]])


#GO = machine_vocab['<go>']
PAD = machine_vocab['<pad>']
EOS = machine_vocab['<eos>']
UNK = machine_vocab['<unk>']


###==== new code
# combined two lists in one list of tubles
def listOfTuples(l1, l2): 
	return list(map(lambda x, y:(x,y), l1, l2)) 


dataset = listOfTuples(text_from, text_to)

###==== end new code


