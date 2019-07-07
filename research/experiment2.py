#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:44:40 2019
Note: keep in mind that when the TURNS windos slides, it should shrink at the end of one conversation rather than keep sliding to include the next conversation!
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

ps = SnowballStemmer('english')

def preprocess(text):
            # Stem and remove stopwords
            text = re.sub('[^a-zA-Z]', ' ', text)
            text = text.lower()
            text = text.split()
            text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
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
TURNS = 1
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
    
    
    

#---------non-overlapping-window----------#
TURNS = 2
tmp = dialogue1['Q'] + ' <eos> ' + dialogue1['A']
Q = []
A = []
for i in range(0, (len(dialogue1)-TURNS), TURNS):
    Q.append(' <eos> '.join(tmp.loc[i:(i+(TURNS-2))]))
    Q[-1] += ' <eos> ' + tmp.loc[i+(TURNS-1)].split(' <eos> ')[0]
    A.append(tmp.loc[i+(TURNS-1)].split(' <eos> ')[1])
    
df1 = pd.DataFrame({'Context':Q, 'Utterance':A})

#repeat with dialogue2
tmp = dialogue2['Q'] + ' <eos> ' + dialogue2['A']
Q = []
A = []
for i in range(0, (len(dialogue2)-TURNS), TURNS):
    Q.append(' <eos> '.join(tmp.loc[i:(i+(TURNS-2))]))
    Q[-1] += ' <eos> ' + tmp.loc[i+(TURNS-1)].split(' <eos> ')[0]
    A.append(tmp.loc[i+(TURNS-1)].split(' <eos> ')[1])

df2 = pd.DataFrame({'Context':Q, 'Utterance':A})

train_df = df1.append(df2, ignore_index = True) 

#repeat with dialogue4p
tmp = dialogue4p['Q'] + ' <eos> ' + dialogue4p['A']
Q = []
A = []
for i in range(0, (len(dialogue4p)-TURNS), TURNS):
    Q.append(' <eos> '.join(tmp.loc[i:(i+(TURNS-2))]))
    Q[-1] += ' <eos> ' + tmp.loc[i+(TURNS-1)].split(' <eos> ')[0]
    A.append(tmp.loc[i+(TURNS-1)].split(' <eos> ')[1])
df3 = pd.DataFrame({'Context':Q, 'Utterance':A})

train_df = train_df.append(df3, ignore_index = True) 

#repeat with dialogue3 for building test df
tmp = dialogue3['Q'] + ' <eos> ' + dialogue3['A']
Q = []
A = []
for i in range(0, (len(dialogue3)-TURNS), TURNS):
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
 
#---------END-OF-non-overlapping-window----------#
    

#------------#
    
#https://github.com/dennybritz/chatbot-retrieval/blob/master/notebooks/TFIDF%20Baseline%20Evaluation.ipynb
    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

y_test = np.zeros(len(test_df))

def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples

def predict_random(context, utterances):
    return np.random.choice(len(utterances), D+1, replace=False)


# Evaluate Random predictor
y_random = [predict_random(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
for n in [1, 3, 5, 302]:
    print("Recall @ ({}, {}): {:g}".format(n, D+1, evaluate_recall(y_random, y_test, n)))
    
    
    
class TFIDFPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values,data.Utterance.values))

    def predict(self, context, utterances):
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]
    
    
    
# Evaluate TFIDF predictor
pred = TFIDFPredictor()
pred.train(train_df)
y = [pred.predict(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
for n in [1, 3, 5, 10, 302]:
    print("Recall @ ({}, {}): {:g}".format(n, D+1, evaluate_recall(y, y_test, n)))

y[-6]
test_df.iloc[-6, 0]
test_df.iloc[-6, 1]
test_df.iloc[-6, 155]
test_df.iloc[-6, 163]
test_df.iloc[-6, 204]

y[10]
test_df.iloc[10, 0]
test_df.iloc[10, 1]
test_df.iloc[10, 77]
test_df.iloc[10, 58]
#-------------------------#
    



train_df.to_csv('/Users/amc/Documents/experiments/chatbot-retrieval/data/train.csv', index=False)
test_df.to_csv('/Users/amc/Documents/experiments/chatbot-retrieval/data/test.csv', index=False)
test_df.to_csv('/Users/amc/Documents/experiments/chatbot-retrieval/data/valid.csv', index=False)












#dataset = pandas.concat([dialogue1, dialogue2, dialogue3])
#Reset index otherwise during the loop below we select multiple rows (pandas.concat results in repeated indices)
#dataset = dataset.reset_index(drop=True)
    
    


querycorpus = []
for i in range(0, len(dataset)):
    query = re.sub('[^a-zA-Z]', ' ', dataset['Q'][i])
    query = query.lower()
    query = query.split()
    query = [ps.stem(word) for word in query if not word in set(stopwords.words('english'))]
    query = ' '.join(query)
    querycorpus.append(query)      

# Creating the Bag of Words model with TFIDF and calc cosine_similarity
vectorizer = CountVectorizer(decode_error="replace")
vec_train = vectorizer.fit_transform(querycorpus) #this is needed to get the attribute vocabulary_
training_vocabulary = vectorizer.vocabulary_
transformer = TfidfTransformer()
trainingvoc_vectorizer = CountVectorizer(decode_error="replace", vocabulary=training_vocabulary)
tfidf_querycorpus = TfidfVectorizer().fit_transform(querycorpus)



def toia_answer(newquery, k=5):

    tfidf_newquery = transformer.fit_transform(trainingvoc_vectorizer.fit_transform(numpy.array([preprocess(newquery)]))) 
    cosine_similarities = cosine_similarity(tfidf_newquery, tfidf_querycorpus)
    related_docs_indices = (-cosine_similarities[0]).argsort()
    sorted_freq = cosine_similarities[0][related_docs_indices]
    
    #note for this distance the problem we had befor with inf, we have now with 0. Again we decide
    #to make the prediction a bit random. This could be adjusted to remove any 0 distance and
    #pick the only ones left if any, and if none predict 1.
    
    if sum(sorted_freq)==0:
        return "Not understood"
    
    elif sorted_freq[k-1]!=sorted_freq[k] or sorted_freq[k-1]==sorted_freq[k]==0:
        selected = related_docs_indices[:k]
       
        return dataset.iloc[selected[0]]['A']
#        return dataset.iloc[selected[0]]['A'], dataset.iloc[selected,:(k-1)]   
#        print("\n Cosine Similarities:", sorted_freq[:k], "\n")
    else:
        indeces = numpy.where(numpy.roll(sorted_freq,1)!=sorted_freq)
        selected = related_docs_indices[:indeces[0][indeces[0]>=k][0]]
    
        return dataset.iloc[selected[0]]['A']
#        return dataset.iloc[selected[0]]['A'], dataset.iloc[selected,:(k-1)]
#        print("\n Cosine Similarities:", sorted_freq[:k], "\n")