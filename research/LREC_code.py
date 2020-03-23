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
from InferSent.encoder.models import InferSent
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

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
%matplotlib inline

# Version
print(mpl.__version__)  #> 3.0.0
print(sns.__version__)  #> 0.9.0

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
train_df.to_csv('MargaritaCorpusKB.csv', encoding='utf-8')

#################

## upload dialogues ##
multiturndialogues = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/DIALOGUES.csv', encoding='utf-8')
#update conversation number so that EDU and PER have different count / conversation ID
multiturndialogues.loc[multiturndialogues.Mode=='EDU', 'Conversation'] = multiturndialogues.loc[multiturndialogues.Mode=='EDU', 'Conversation'] + 10

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print('--------- KB Categories ------------ \n # utterances = {}'.format( pd.crosstab(index=train_df["Category"], columns="count") ))

print('--------- KB TRAIN ------------ \n # utterances = {}'.format(len(train_df.Utterance.values)))
print('# words = {}'.format(sum([len(tokens) for tokens in [utt.split() for utt in train_df.Utterance.values]])
      + sum([len(tokens) for tokens in [utt.split() for utt in train_df.Context.values]])))
print('avg. words per utterance = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in train_df.Utterance.values]])))
print('avg. words per context = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in train_df.Context.values]])))
print('# unique questions = {}'.format(len(np.unique(train_df.Context))))
print('# unique answers = {}'.format(len(np.unique(train_df.Utterance))))

print('--------- DIALOGUES All ------------ \n # utterances = {}'.format(len(multiturndialogues.A.values)))
print('# words = {}'.format(sum([len(tokens) for tokens in [utt.split() for utt in multiturndialogues.A.values]])
      + sum([len(tokens) for tokens in [utt.split() for utt in multiturndialogues.Q.values]])))
print('avg. words per utterance = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in multiturndialogues.A.values]])))
print('avg. words per context = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in multiturndialogues.Q.values]])))
conversation_IDs = np.unique(multiturndialogues.Conversation)
print('min # turns per dialogue = {}'.format(min([len(multiturndialogues.loc[multiturndialogues.Conversation==i]) for i in conversation_IDs])))      
print('avg # turns per dialogue = {}'.format(np.mean([len(multiturndialogues.loc[multiturndialogues.Conversation==i]) for i in conversation_IDs])))

subset = multiturndialogues.loc[multiturndialogues.Experiment.isin(['TRAIN'])]
print('--------- DIALOGUES TRAIN ------------ \n # utterances = {}'.format(len(subset.A.values)))
print('# words = {}'.format(sum([len(tokens) for tokens in [utt.split() for utt in subset.A.values]])
      + sum([len(tokens) for tokens in [utt.split() for utt in subset.Q.values]])))
print('avg. words per utterance = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in subset.A.values]])))
print('avg. words per context = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in subset.Q.values]])))
conversation_IDs = np.unique(subset.Conversation)
print('min # turns per dialogue = {}'.format(min([len(subset.loc[subset.Conversation==i]) for i in conversation_IDs])))      
print('avg # turns per dialogue = {}'.format(np.mean([len(subset.loc[subset.Conversation==i]) for i in conversation_IDs])))
      
subset = multiturndialogues.loc[multiturndialogues.Experiment.isin(['TEST'])]
print('--------- DIALOGUES TEST ------------ \n # utterances = {}'.format(len(subset.A.values)))
print('# words = {}'.format(sum([len(tokens) for tokens in [utt.split() for utt in subset.A.values]])
      + sum([len(tokens) for tokens in [utt.split() for utt in subset.Q.values]])))
print('avg. words per utterance = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in subset.A.values]])))
print('avg. words per context = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in subset.Q.values]])))
conversation_IDs = np.unique(subset.Conversation)
print('min # turns per dialogue = {}'.format(min([len(subset.loc[subset.Conversation==i]) for i in conversation_IDs])))      
print('avg # turns per dialogue = {}'.format(np.mean([len(subset.loc[subset.Conversation==i]) for i in conversation_IDs])))
      
subset = multiturndialogues.loc[multiturndialogues.Mode.isin(['EDU'])]
print('--------- DIALOGUES UNI MODE ------------ \n # utterances = {}'.format(len(subset.A.values)))
print('# words = {}'.format(sum([len(tokens) for tokens in [utt.split() for utt in subset.A.values]])
      + sum([len(tokens) for tokens in [utt.split() for utt in subset.Q.values]])))
print('avg. words per utterance = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in subset.A.values]])))
print('avg. words per context = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in subset.Q.values]])))
conversation_IDs = np.unique(subset.Conversation)
print('min # turns per dialogue = {}'.format(min([len(subset.loc[subset.Conversation==i]) for i in conversation_IDs])))      
print('avg # turns per dialogue = {}'.format(np.mean([len(subset.loc[subset.Conversation==i]) for i in conversation_IDs])))
      
subset = multiturndialogues.loc[multiturndialogues.Mode.isin(['PER'])]
print('--------- DIALOGUES PERS MODE ------------ \n # utterances = {}'.format(len(subset.A.values)))
print('# words = {}'.format(sum([len(tokens) for tokens in [utt.split() for utt in subset.A.values]])
      + sum([len(tokens) for tokens in [utt.split() for utt in subset.Q.values]])))
print('avg. words per utterance = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in subset.A.values]])))
print('avg. words per context = {}'.format(np.mean([len(tokens) for tokens in [utt.split() for utt in subset.Q.values]])))
conversation_IDs = np.unique(subset.Conversation)
print('min # turns per dialogue = {}'.format(min([len(subset.loc[subset.Conversation==i]) for i in conversation_IDs])))      
print('avg # turns per dialogue = {}'.format(np.mean([len(subset.loc[subset.Conversation==i]) for i in conversation_IDs])))
      

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


def add_eos(list_of_docs):
    return ' [SEP] '.join(list_of_docs)


def train_test_sets_questions_seqdata_no_ooc(multiturndialogues, TURNS = 1):
#    hold_out_sample = np.array([1, 2]) HAD TO CHANGE THIS because when building the test set, the WOz "did not see" the answer from other test set conversations, so the test set can only be 1 conversation. And even, this way it's tricky as the WoZ recycled in some cases the answers from the same conversation.
    
    
    if TURNS==0:
        #TRAIN
        train_df = multiturndialogues.loc[multiturndialogues.Experiment.isin(['TRAIN'])][['Q','A']]
        train_df.reset_index(level=None, drop=True, inplace=True)
        train_df = train_df.rename(columns = {"Q":"Context", "A": "Utterance"})
        #TEST
        test_df = multiturndialogues.loc[multiturndialogues.Experiment.isin(['TEST'])]
        Context, WOzAnswersIDs = [], []
        allAs = list(train_df.Utterance.values)
        for example_id in range(len(test_df)):
            exampleWOzAnswers = list(test_df.iloc[example_id, 5:].values)
#            if not allnull(exampleWOzAnswers):
            tmp_WAs = []
            for distr in allAs:
                if distr in exampleWOzAnswers:
                    tmp_WAs.append(allAs.index(distr))
            Context.append(test_df.iloc[example_id, 1])
            WOzAnswersIDs.append(tmp_WAs)
    
    
    
    if TURNS>0:
        #TRAIN
        train_df = pd.DataFrame({'Context':[], 'Utterance':[]})
        train_samples = np.unique(multiturndialogues.loc[multiturndialogues.Experiment=='TRAIN', 'Conversation'])
        test_samples = np.unique(multiturndialogues.loc[multiturndialogues.Experiment=='TEST', 'Conversation'])
        
        for dial in train_samples:
            #re-write this step: here we should just select a list of sequential Q-A according to par TURNS then combine them with the add_eos fn.
            tmp = multiturndialogues.loc[multiturndialogues.Conversation==dial]['Q'] + ' [SEP] ' + multiturndialogues.loc[multiturndialogues.Conversation==dial]['A']
            tmp.reset_index(level=None, drop=True, inplace=True)
            Q, A = [], []
            for i in range(len(tmp)-TURNS-1):
                Q.append(add_eos(tmp.loc[i:(i+(TURNS-1))]))
                Q[-1] += ' [SEP] ' + tmp.loc[i+TURNS].split(' [SEP] ')[0]
                A.append(tmp.loc[i+(TURNS)].split(' [SEP] ')[1])
            df = pd.DataFrame({'Context':Q, 'Utterance':A})
            train_df = train_df.append(df, ignore_index = True)             
        #TEST      
        for dial in test_samples:
            tmp = multiturndialogues.loc[multiturndialogues.Conversation==dial]['Q'] + ' [SEP] ' + multiturndialogues.loc[multiturndialogues.Conversation==dial]['A']
            tmp.reset_index(level=None, drop=True, inplace=True)
            tmp_WozAnswers = multiturndialogues.loc[multiturndialogues.Conversation==dial]
            tmp_WozAnswers.reset_index(level=None, drop=True, inplace=True)
            
            Context, WOzAnswersIDs = [], []
            allAs = list(train_df.Utterance.values)
            for i in range(len(tmp)-TURNS-1):
                Context.append(add_eos(tmp.loc[i:(i+(TURNS-1))]))
                Context[-1] += ' [SEP] ' + tmp.loc[i+TURNS].split(' [SEP] ')[0]
                ##################
                exampleWOzAnswers = list(tmp_WozAnswers.iloc[i+TURNS, 7:].values)
#                if not allnull(exampleWOzAnswers):
                tmp_WAs = []
                for distr in allAs:
                    if distr in exampleWOzAnswers:
                        tmp_WAs.append(allAs.index(distr))
                WOzAnswersIDs.append(tmp_WAs)

                    
    test_df = pd.DataFrame({'Context':Context, 'WOzAnswers':WOzAnswersIDs})        
                
        
    return train_df, test_df

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
    num_examples = float(len(y))
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
            num_correct_ans += len(list(intersection))
        elif len(intersection)==0 and len(A)==0 and len(B)==0:
            num_correct_nonans += 1
    return (num_correct_ans+num_correct_nonans)/num_examples, num_correct_ans, num_correct_nonans


class TFIDFPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values, data.Utterance.values))

    def predict(self, context, utterances):
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
#        return np.argsort(result, axis=0)[::-1]
        return result

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
#    return np.linalg.norm(u-v)


def INFERSENTPredictor_new(context, utterances):
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


#-- EOfn --#  ###NOTE: need to add to the dialogues TRAIN the wizard answers



valid_df = multiturndialogues.loc[multiturndialogues.Experiment.isin(["TRAIN"])]
valid_df.reset_index(level=None, drop=True, inplace=True)
valid_df = test_set_questions_ooctrain(valid_df, train_df)

test_df = multiturndialogues.loc[multiturndialogues.Experiment.isin(["TEST"])]
test_df.reset_index(level=None, drop=True, inplace=True)        
test_df = test_set_questions_ooctrain(test_df, train_df)


train_nonans = sum([len(ls)==0 for ls in valid_df.WOzAnswers.values])
train_ans = sum([len(ls)!=0 for ls in valid_df.WOzAnswers.values])
prop_train_nonans = train_nonans/(train_nonans + train_ans)
print('#Dialogue TRAIN utterances = {}'.format(len(valid_df.Context.values)))
print(
      prop_train_nonans,
      train_nonans,
      train_ans 
      )

prop_test_nonans = sum([len(ls)==0 for ls in test_df.WOzAnswers.values])/len(test_df.WOzAnswers.values)
print('#Dialogue TEST utterances = {}'.format(len(test_df.Context.values)))
print(
      prop_test_nonans,
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in test_df.WOzAnswers.values])
      )

#upsample valid data to achieve same prop of prop_test_nonans
upsample_nonans = int((train_nonans - prop_test_nonans*(train_nonans + train_ans))/(prop_test_nonans - 1))
#get all answer indices in valid_df
indeces=[]
for ls in valid_df.WOzAnswers.values:
    indeces += ls
indeces=np.unique(indeces) #(useless but elegant)
#subset where answers are not in train dialogues (might want to double check repetitions...)
answers_in_valid_df = np.unique(train_df.loc[train_df.index.isin(indeces), 'Utterance'])
subset = train_df.loc[~train_df.Utterance.isin(answers_in_valid_df)]
#sample
subset = subset.sample(n=upsample_nonans, replace=False, random_state=1985)
#remove from train (note there is 1 repeated qs - check len of questions_in_sample vs len of sample)
questions_in_sample = np.unique(subset.Context.values)
train_df = train_df.loc[~train_df.Context.isin(questions_in_sample)]
train_df = train_df.reset_index()

#reset index of train and rebuild valid_df, test_df to pick up the right indices
valid_df = multiturndialogues.loc[multiturndialogues.Experiment.isin(["TRAIN"])]
valid_df.reset_index(level=None, drop=True, inplace=True)
valid_df = test_set_questions_ooctrain(valid_df, train_df)
#and add samples to valid_df
subset = pd.DataFrame({'Context':subset.Context.values, 'WOzAnswers':[ [] for i in range(len(subset.Context.values))]})
valid_df = valid_df.append(subset, ignore_index=True)
valid_df.reset_index()

test_df = multiturndialogues.loc[multiturndialogues.Experiment.isin(["TEST"])]
test_df.reset_index(level=None, drop=True, inplace=True)        
test_df = test_set_questions_ooctrain(test_df, train_df)

train_nonans = sum([len(ls)==0 for ls in valid_df.WOzAnswers.values])
train_ans = sum([len(ls)!=0 for ls in valid_df.WOzAnswers.values])

prop_train_nonans = train_nonans/(train_nonans + train_ans)
print('#Dialogue TRAIN utterances = {}'.format(len(valid_df.Context.values)))
print(
      prop_train_nonans,
      train_nonans,
      train_ans 
      )
prop_test_nonans = sum([len(ls)==0 for ls in test_df.WOzAnswers.values])/len(test_df.WOzAnswers.values)
print('#Dialogue TEST utterances = {}'.format(len(test_df.Context.values)))
print(
      prop_test_nonans,
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in test_df.WOzAnswers.values])
      )

#### NOTE Train data (KB) - we should strip off the samples we inserted in the valid_df. Moreover, we shall calc centroids for questions corresponding to answers that appear in valid_df vs. questions relative to answers that were not selected 


# Train TFIDF predictor
pred = TFIDFPredictor()
pred.train(train_df)

y = [pred.predict(valid_df.Context[x], list(train_df.Context.values)) for x in range(len(valid_df))]
ans = []
nonans = []
thresholds = []
for thr in np.arange(0.0, 1.0, 0.05):
    ans.append(evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[1])
    nonans.append(evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[2])
    thresholds.append(thr)
pred_thr = thresholds[np.argsort([sum(x) for x in zip([l/train_ans*(1-prop_train_nonans) for l in ans], [l/train_nonans*prop_train_nonans for l in nonans])], axis=0)[::-1][0]]

# Evaluate TFIDF predictor QUESTION SIMILARITY (use train_df.Utterance.values for ANSWER SIM)
for i in range(3):
    for thr in np.arange(0.0, 1.0, 0.05):
        print("Recall@1 for thr={}: {:g}".format(thr, evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[i]))
# Test TFIDF predictor
y = [pred.predict(test_df.Context[x], list(train_df.Context.values)) for x in range(len(test_df))]
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, test_df.WOzAnswers.values, n, thr=pred_thr)[i]))
        
## Evaluate TFIDF predictor Q-A SIMILARITY
#y = [pred.predict(valid_df.Context[x], list(train_df.Utterance.values)) for x in range(len(valid_df))]
#for i in range(3):
#    for thr in np.arange(0.0, 1.0, 0.05):
#        print("Recall@1 for thr={}: {:g}".format(thr, evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[i]))
## Test TFIDF predictor
#y = [pred.predict(test_df.Context[x], list(train_df.Utterance.values)) for x in range(len(test_df))]
#for i in range(3):
#    for n in [1, 2, 5, 10, 20]:
#        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, test_df.WOzAnswers.values, n, thr=0.25)[i]))

for testex in range(10):
    tmp = np.argsort(y[testex ], axis=0)[::-1][:3]
    print(tmp)
    print(np.sort(y[testex], axis=0)[::-1][:3])
    
    Qid = tmp[0]
    print("Test Question: ", test_df.Context[testex ])
    print("Train Question: ", train_df.Context[Qid])

q = []
GA = []
A1 = []
A2 = []
A3 = []
A4 = []
A5 = []
for testex in range(len(test_df)):
    question = test_df.Context[testex]
    Qids = np.argsort(y[testex], axis=0)[::-1][:5]
    As = train_df.Utterance[Qids]
    RANKs = np.sort(y[testex], axis=0)[::-1][:5]
    q.append(question)
    GA.append(multiturndialogues[multiturndialogues.Q == question].A.iloc[0])
    A1.append(As.iloc[0] + " (rank={})".format(RANKs[0]))
    A2.append(As.iloc[1] + " (rank={})".format(RANKs[1]))
    A3.append(As.iloc[2] + " (rank={})".format(RANKs[2]))
    A4.append(As.iloc[3] + " (rank={})".format(RANKs[3]))
    A5.append(As.iloc[4] + " (rank={})".format(RANKs[4]))
TFIDFdialoguesout = pd.DataFrame({'q':q, 'GA':GA, '1A':A1, '2A':A2, '3A':A3, '4A':A4, '5A':A5})   
TFIDFdialoguesout.to_csv('TF-IDF dialogues.csv', encoding='utf-8')   


### InferSent ###
model_version = 1
MODEL_PATH = "InferSent/encoder/infersent%s.pickle" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'InferSent/dataset/GloVe/glove.840B.300d.txt' if model_version == 1 else 'InferSent/dataset/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

allsentences = []
for item in train_df.Context.values: allsentences.append(item)
for item in train_df.Utterance.values: allsentences.append(item) # used answer too to expand vocabulary
model.build_vocab(allsentences, tokenize=True)

#QUESTION SIMILARITY
train_embeddings = model.encode(train_df.Context.values, bsize=128, tokenize=False, verbose=True)
valid_embeddings = model.encode(valid_df.Context.values, bsize=128, tokenize=False, verbose=True)
test_embeddings = model.encode(test_df.Context.values, bsize=128, tokenize=False, verbose=True)

##Examples mentioned in paper: Q id 20 and 30, answ id 2017 and 77 when the test set was conversations 1 and 2, and I kept empty WoZ answers
#np.argsort([cosine(valid_embeddings[1], utt) for utt in train_embeddings], axis=0)[::-1][:5]
#np.sort([cosine(valid_embeddings[1], utt) for utt in train_embeddings], axis=0)[::-1][:5]
#valid_df.Context[1]
#valid_df.WOzAnswers[1]
#train_df.Context[133]
#train_df.Utterance[133]

### I had to put the WOZ answers given by picking from new data to properly evaluate this. --> think if it makes sense to include the ooc answers too?

# Evaluate InferSent predictor
y = [INFERSENTPredictor_new(valid_embeddings[x], train_embeddings) for x in range(len(valid_embeddings))]
ans = []
nonans = []
thresholds = []
for thr in np.arange(0.0, 1.0, 0.05):
    ans.append(evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[1])
    nonans.append(evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[2])
    thresholds.append(thr)
pred_thr = thresholds[np.argsort([sum(x) for x in zip([l/train_ans*(1-prop_train_nonans) for l in ans], [l/train_nonans*prop_train_nonans for l in nonans])], axis=0)[::-1][0]]

for i in range(3):
    for thr in np.arange(0, 1, 0.05):
        print("Recall@1 for thr={}: {:g}".format(thr, evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[i]))

# Test InferSent predictor
y = [INFERSENTPredictor_new(test_embeddings[x], train_embeddings) for x in range(len(test_embeddings))]
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, test_df.WOzAnswers.values, n, thr=pred_thr)[i]))

for testex in range(10):
    tmp = np.argsort(y[testex ], axis=0)[::-1][:3]
    print(tmp)
    print(np.sort(y[testex], axis=0)[::-1][:3])
    
    Qid = tmp[0]
    print("Test Question: ", test_df.Context[testex ])
    print("Train Question: ", train_df.Context[Qid])
    
print("InferSent Dialogue")    
for testex in range(34):
    Qid = np.argsort(y[testex ], axis=0)[::-1][0]
    print("Question (Test Set): ", test_df.Context[testex])
    print("Answer (KB'): ", train_df.Utterance[Qid])
    print("(Cosine Similarity: {})".format(np.sort(y[testex], axis=0)[::-1][0]))



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
for text in train_df.Context.values:
    train_embeddings.append(bertembed(text)[0])
train_embeddings = np.array(train_embeddings)
valid_embeddings = []
for text in valid_df.Context.values:
    valid_embeddings.append(bertembed(text)[0])
valid_embeddings = np.array(valid_embeddings)
test_embeddings = []
for text in test_df.Context.values:
    test_embeddings.append(bertembed(text)[0])
test_embeddings = np.array(test_embeddings)
 
y = [INFERSENTPredictor_new(valid_embeddings[x], train_embeddings) for x in range(len(valid_embeddings))]
ans = []
nonans = []
thresholds = []
for thr in np.arange(0.0, 1.0, 0.05):
    ans.append(evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[1])
    nonans.append(evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[2])
    thresholds.append(thr)
pred_thr = thresholds[np.argsort([sum(x) for x in zip([l/train_ans*(1-prop_train_nonans) for l in ans], [l/train_nonans*prop_train_nonans for l in nonans])], axis=0)[::-1][0]]

for i in range(3):
    for thr in np.arange(0, 1, 0.05):
        print("Recall@1 for thr={}: {:g}".format(thr, evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[i]))

y = [INFERSENTPredictor_new(test_embeddings[x], train_embeddings) for x in range(len(test_embeddings))]
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, test_df.WOzAnswers.values, n, thr=pred_thr)[i]))
#MEDIAN pooling better than mean and close to TFIDF on test set --need more error analysis because HP: BERT does better selection than tfidf even if wrong question (second best predition by bert better than second best by tfidf). Mode pretty shit. Max better than mode, worse than mean. mean-max not great too. Bert cased doesn't make it better. speach in practice will be converted to lowercase text anyway. Large BERT doesn't bring improvements


for testex in range(10):
    tmp = np.argsort(y[testex], axis=0)[::-1][:3]
    print(tmp)
    print(np.sort(y[testex], axis=0)[::-1][:3])
    
    Qid = tmp[0]
    print("Test Question: ", test_df.Context[testex])
    print("Train Question: ", train_df.Context[Qid])

q = []
GA = []
A1 = []
A2 = []
A3 = []
A4 = []
A5 = []
for testex in range(len(test_df)):
    question = test_df.Context[testex]
    Qids = np.argsort(y[testex], axis=0)[::-1][:5]
    As = train_df.Utterance[Qids]
    RANKs = np.sort(y[testex], axis=0)[::-1][:5]
    q.append(question)
    GA.append(multiturndialogues[multiturndialogues.Q == question].A.iloc[0])
    A1.append(As.iloc[0] + " (rank={})".format(RANKs[0]))
    A2.append(As.iloc[1] + " (rank={})".format(RANKs[1]))
    A3.append(As.iloc[2] + " (rank={})".format(RANKs[2]))
    A4.append(As.iloc[3] + " (rank={})".format(RANKs[3]))
    A5.append(As.iloc[4] + " (rank={})".format(RANKs[4]))
BERTdialoguesout = pd.DataFrame({'q':q, 'GA':GA, '1A':A1, '2A':A2, '3A':A3, '4A':A4, '5A':A5})   
BERTdialoguesout.to_csv('BERT dialogues.csv', encoding='utf-8')    
    
# ANSWER SIMILARITY
train_embeddings = []
for text in train_df.Utterance.values:
    train_embeddings.append(bertembed(text)[0])
train_embeddings = np.array(train_embeddings)
 
y = [INFERSENTPredictor_new(valid_embeddings[x], train_embeddings) for x in range(len(valid_embeddings))]
for i in range(3):
    for thr in np.arange(0, 1, 0.05):
        print("Recall@1 for thr={}: {:g}".format(thr, evaluate_recall_thr(y, valid_df.WOzAnswers.values, k=1, thr=thr)[i]))
    
y = [INFERSENTPredictor_new(test_embeddings[x], train_embeddings) for x in range(len(test_embeddings))]
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, test_df.WOzAnswers.values, n, thr=0.7)[i]))


#centroids
KBcentroid = train_embeddings.mean(0)
dist_to_KBcentr = [cosine(valid_embeddings[x], KBcentroid) for x in range(len(valid_embeddings))]
# EUCLIDEAN DISTANCE np.linalg.norm(u-v)
#dist_to_KBcentr = [np.linalg.norm(valid_embeddings[x]-KBcentroid) for x in range(len(valid_embeddings))]
valid_df['dist_to_KBcentr'] = dist_to_KBcentr 
ans_label = ['HAVE ans' if len(valid_df.WOzAnswers.values[x])>0 else 'NO ans' for x in range(len(valid_embeddings))]
valid_df['ans_label'] = ans_label

# Draw Plot
plt.figure(figsize=(8,5), dpi= 70)
sns.kdeplot(valid_df.loc[valid_df['ans_label'] == 'HAVE ans', 'dist_to_KBcentr'], shade=True, color="g", label="Have Ans", alpha=.7)
sns.kdeplot(valid_df.loc[valid_df['ans_label'] == 'NO ans', 'dist_to_KBcentr'], shade=True, color="deeppink", label="No ans", alpha=.7)

# Decoration
plt.title('Density Plot of Cosine Distance to KB Centroid \n (BERT)', fontsize=22)
plt.legend()
plt.show()
  

#####TEST
##centroids
#KBcentroid = train_embeddings.mean(0)
#dist_to_KBcentr = [cosine(test_embeddings[x], KBcentroid) for x in range(len(test_embeddings))]
#test_df['dist_to_KBcentr'] = dist_to_KBcentr 
#ans_label = ['HAVE ans' if len(test_df.WOzAnswers.values[x])>0 else 'NO ans' for x in range(len(test_embeddings))]
#test_df['ans_label'] = ans_label
#
## Draw Plot
#plt.figure(figsize=(16,10), dpi= 80)
#sns.kdeplot(test_df.loc[test_df['ans_label'] == 'HAVE ans', 'dist_to_KBcentr'], shade=True, color="g", label="Have Ans", alpha=.7)
#sns.kdeplot(test_df.loc[test_df['ans_label'] == 'NO ans', 'dist_to_KBcentr'], shade=True, color="deeppink", label="No ans", alpha=.7)
#
## Decoration
#plt.title('Density Plot of Cosine Distance to KB Centroid by have/not have ans.', fontsize=22)
#plt.legend()
#plt.show()




###### using dialogue turns ##################
    
    
train_df, test_df = train_test_sets_questions_seqdata_no_ooc(multiturndialogues, TURNS=0)
print('TRAIN # utterances = {}'.format(len(train_df.Context.values)))
print('TEST # utterances = {}'.format(len(test_df.Context.values)))
print(
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values])/len(test_df.WOzAnswers.values),
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in test_df.WOzAnswers.values]) 
      )


train_df, test_df = train_test_sets_questions_seqdata_no_ooc(multiturndialogues, TURNS=1)
print('TRAIN # utterances = {}'.format(len(train_df.Context.values)))
print('TEST # utterances = {}'.format(len(test_df.Context.values)))
print(
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values])/len(test_df.WOzAnswers.values),
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in test_df.WOzAnswers.values]) 
      )


  ### Seems there are bugs to fix here ###  
valid_df = multiturndialogues.loc[multiturndialogues.Experiment.isin(["TRAIN"])]
valid_df.reset_index(level=None, drop=True, inplace=True)
valid_df = test_set_questions_ooctrain(valid_df, train_df)

test_df = multiturndialogues.loc[multiturndialogues.Experiment.isin(["TEST"])]
test_df.reset_index(level=None, drop=True, inplace=True)        
test_df = test_set_questions_ooctrain(test_df, train_df)

print('# utterances = {}'.format(len(valid_df.Context.values)))
print(
      sum([len(ls)==0 for ls in valid_df.WOzAnswers.values])/len(valid_df.WOzAnswers.values),
      sum([len(ls)==0 for ls in valid_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in valid_df.WOzAnswers.values]) 
      )

print('# utterances = {}'.format(len(test_df.Context.values)))
print(
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values])/len(test_df.WOzAnswers.values),
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in test_df.WOzAnswers.values])
      )


train_df, test_df = train_test_sets_questions_seqdata_no_ooc(multiturndialogues, TURNS=0)
print('TRAIN # utterances = {}'.format(len(train_df.Context.values)))
print('TEST # utterances = {}'.format(len(test_df.Context.values)))
print(
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values])/len(test_df.WOzAnswers.values),
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in test_df.WOzAnswers.values]) 
      )


train_df, test_df = train_test_sets_questions_seqdata_no_ooc(multiturndialogues, TURNS=1)
print('TRAIN # utterances = {}'.format(len(train_df.Context.values)))
print('TEST # utterances = {}'.format(len(test_df.Context.values)))
print(
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values])/len(test_df.WOzAnswers.values),
      sum([len(ls)==0 for ls in test_df.WOzAnswers.values]),
      sum([len(ls)!=0 for ls in test_df.WOzAnswers.values]) 