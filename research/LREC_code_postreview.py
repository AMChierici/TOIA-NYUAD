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
# train_df.to_csv('MargaritaCorpusKB.csv', encoding='utf-8')

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



valid_df = multiturndialogues.loc[multiturndialogues.Experiment.isin(["TRAIN"]) & multiturndialogues.Mode.isin(["PER"])]
valid_df.reset_index(level=None, drop=True, inplace=True)
valid_df = test_set_questions_ooctrain(valid_df, train_df)

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
  


########## USING BERT ##########
model_path = '/Users/amc/Documents/fine_tuned_models/bert_text_classification/Margarita_1to100/'
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(model_path)
# Load pre-trained model (weights)
model = BertModel.from_pretrained(model_path)
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

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

    
# ANSWER SIMILARITY
train_embeddings = []
for text in train_df.Utterance.values:
    train_embeddings.append(bertembed(text)[0])
train_embeddings = np.array(train_embeddings)
 
y = [LMPredictor_new(valid_embeddings[x], train_embeddings) for x in range(len(valid_embeddings))]
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, valid_df.WOzAnswers.values, n, thr=0)[i]))




# qarelevance
preds = pd.read_csv('/Users/amc/Documents/glue_data/Margarita_1_100_ratio/valid_results_mrpc_proba.txt', sep='\t', encoding='utf-8')['prediction'].values
###DO run toia_data_processor.py until row 166
valid_preds = pd.DataFrame({'q': valid_df['#1 String'].values, 'A': valid_df['#2 String'].values, 'y_pred': preds})
###DO re-run this script until row 408
y = []
for i in range(len(valid_df)):
    ranks=[]
    for j in train_corpus:
        answers = train_df[train_df['Context']==j]['Utterance'].values[0]
        rank = valid_preds[(valid_preds['A']==answers) & (valid_preds['q']==valid_df.loc[i, 'Q'])]['y_pred'].values[0]
        ranks.append(rank)
    y.append(ranks)
    
valid_df = test_set_questions_ooctrain(valid_df, train_df)

for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, valid_df.WOzAnswers.values, n, thr=0)[i]))

saveJsonDialogues('data/devBERTqaRel1to100Dialogues.json')


for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y, valid_df.WOzAnswers.values, n, thr=0)[i]))

outputPred(y, valid_df.WOzAnswers.values, train_df.Utterance.values, '/Users/amc/Documents/TOIA-NYUAD/research/data/devBERTqaRel1to100Results.csv', 20, valid_df.Context.values)
#note that if instead of train_df.Utterance.values we use train_corpus or train_df.Context.values (which is the same thing), we get the questions instead of the answers
        
#RANDOM
import random as rd
tmp = list(range(1,1+len(train_corpus)))
y_random = [rd.sample(tmp, len(tmp)) for x in range(len(valid_df))] 
for i in range(3):
    for n in [1, 2, 5, 10, 20]:
        print("Recall@{}: {:g}".format(n, evaluate_recall_thr(y_random, valid_df.WOzAnswers.values, n, thr=0)[i]))

 
####### Using DNN initialized on ni sentence encoders model + BM25 ######
def print_metrics(y):                         
    query_precisions, query_recalls, query_precision_at20s, query_recall_at20s, query_precision_at10s, query_recall_at10s,  query_precision_at5s, query_recall_at5s,  query_precision_at2s, query_recall_at2s,  query_precision_at1s, query_recall_at1s, ave_precisions, rec_ranks, check = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for query, retrieval_scores in zip(list(valid_df.Q.values), y):
        # documents=corpus
        sorted_retrieval_scores = np.sort(retrieval_scores, axis=0)[::-1]
        if sorted_retrieval_scores[0]==0:
            sorted_retrieved_documents = []
            relevant_documents = []
            query_precisions.append(0)
            query_recalls.append(0)
            query_precision_at20s.append(0)
            query_precision_at10s.append(0)
            query_precision_at5s.append(0)
            query_precision_at2s.append(0)
            query_precision_at1s.append(0)
            query_recall_at20s.append(0)
            query_recall_at10s.append(0)
            query_recall_at5s.append(0)
            query_recall_at2s.append(0)
            query_recall_at1s.append(0)
            ave_precisions.append(0)
            rec_ranks.append(0)
        else:
            sorted_id_documents = np.argsort(retrieval_scores, axis=0)[::-1]
            sorted_id_retreved_documents = sorted_id_documents[sorted_retrieval_scores>0]
            sorted_retrieved_documents = [KBanswers[i] for i in sorted_id_retreved_documents]
            relevant_documents = list(valid_df[['BA1', 'BA2', 'BA3', 'BA4', 'BA5', 'BA6']].loc[valid_df.Q.values==query].values[0])
            # relevant_documents = list(train_df[train_df['Utterance'].isin(relevant_answers)].Context)    
            query_precisions.append(len(set(relevant_documents) & set(sorted_retrieved_documents)) / len(set(sorted_retrieved_documents)))
            query_recalls.append(len(set(relevant_documents) & set(sorted_retrieved_documents)) / len(set(relevant_documents)))    
            query_precision_at20s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:20])) / len(set(sorted_retrieved_documents[:20])))
            query_precision_at10s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:10])) / len(set(sorted_retrieved_documents[:10])))
            query_precision_at5s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:5])) / len(set(sorted_retrieved_documents[:5])))
            query_precision_at2s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:2])) / len(set(sorted_retrieved_documents[:2])))
            query_precision_at1s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:1])) / len(set(sorted_retrieved_documents[:1])))
            # query_recall_at20s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:20])) / min(len(set(relevant_documents)), 20))
            # query_recall_at10s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:10])) / min(len(set(relevant_documents)), 10))
            # query_recall_at5s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:5])) / min(len(set(relevant_documents)), 5))
            # query_recall_at2s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:2])) / min(len(set(relevant_documents)), 2))
            query_recall_at1s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:1])) / min(len(set(relevant_documents)), 1))
            p_at_ks, rel_at_ks = [], []
            for k in range(1, 1+len(set(sorted_retrieved_documents))):
                p_at_ks.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:k])) / len(set(sorted_retrieved_documents[:k])))
                rel_at_ks.append(1 if sorted_retrieved_documents[k-1] in relevant_documents else 0)
            ave_precisions.append(sum([p*r for p, r in zip(p_at_ks, rel_at_ks)])/len(set(relevant_documents)))
            if query_recalls[-1]>0:
                for r, doc in enumerate(sorted_retrieved_documents):
                    if doc in relevant_documents:
                        rec_ranks.append(1/(1+r))
                        break
                else:
                    rec_ranks.append(0)
        check.append([query, sorted_retrieved_documents[:3], relevant_documents])
            
    print("Mean Average Precision (MAP): ", np.mean(ave_precisions))
    print("Mean Reciprocal Rank (MRR): ", np.mean(rec_ranks))
    print("Mean precision: ", np.mean(query_precisions))
    print("Mean recall: ", np.mean(query_recalls))
    print("Mean precision @20: ", np.mean(query_precision_at20s))
    print("Mean precision @10: ", np.mean(query_precision_at10s))
    print("Mean precision @5: ", np.mean(query_precision_at5s))
    print("Mean precision @2: ", np.mean(query_precision_at2s))
    print("Mean precision @1: ", np.mean(query_precision_at1s))
    # print("Mean recall @20: ", np.mean(query_recall_at20s))
    # print("Mean recall @10: ", np.mean(query_recall_at10s))
    # print("Mean recall @5: ", np.mean(query_recall_at5s))
    # print("Mean recall @2: ", np.mean(query_recall_at2s))
    print("Mean recall @1: ", np.mean(query_recall_at1s))

model = tf.keras.models.load_model('/Users/amc/Documents/TOIA-NYUAD/research/saved_dl_unisentencqa_sensqa_model1/model')  
model.summary()    
module = hub.load('./3')

KBanswers = list(np.unique(train_df.Utterance.values))
input_text = 'Is this robot artificial intelligence connected to the work, right?'
def predict_rankings(input_text):
    predictions = []
    for A in KBanswers:
        sentence = input_text + " " + A
        predictions.append(sentence)
    embeddings = module.signatures['question_encoder'](tf.constant(predictions))
    embeddings=np.array(embeddings['outputs'])
    return model.predict(embeddings)

rankings = predict_rankings(input_text)

### This is ONLY DNN ### (BTW this y takes a long to compute... need to optmize code/method here!)
y = [predict_rankings(query) for query in list(valid_df.Q.values)] 
yhat = [[i[0] for i in y[j].tolist()] for j in range(len(y))]
print_metrics(yhat)

# Then, let's see if we can improve by combining DNN and BM25
# k=50

def print_metrics(y, yhat):                         
    query_precisions, query_recalls, query_precision_at20s, query_recall_at20s, query_precision_at10s, query_recall_at10s,  query_precision_at5s, query_recall_at5s,  query_precision_at2s, query_recall_at2s,  query_precision_at1s, query_recall_at1s, ave_precisions, rec_ranks, check = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for query, retrieval_scores, answer_scores in zip(list(valid_df.Q.values), y, yhat):
        # documents=corpus
        if np.sum(retrieval_scores)==0:
            sorted_retrieved_documents = []
            relevant_documents = []
            query_precisions.append(0)
            query_recalls.append(0)
            query_precision_at20s.append(0)
            query_precision_at10s.append(0)
            query_precision_at5s.append(0)
            query_precision_at2s.append(0)
            query_precision_at1s.append(0)
            query_recall_at20s.append(0)
            query_recall_at10s.append(0)
            query_recall_at5s.append(0)
            query_recall_at2s.append(0)
            query_recall_at1s.append(0)
            ave_precisions.append(0)
            rec_ranks.append(0)
        else:
            for i in range(len(train_corpus)):
                ans = train_df[train_df['Context']==(train_corpus[i])].Utterance.values
                ids =  [j for j,x in enumerate(KBanswers) if x in ans]
                scores = [answer_scores[m] for m in ids]
                score = np.mean(scores)
                retrieval_scores[i] = retrieval_scores[i] + score  
            sorted_retrieval_scores = np.sort(retrieval_scores, axis=0)[::-1]
            sorted_id_documents = np.argsort(retrieval_scores, axis=0)[::-1]
            sorted_id_retreved_documents = sorted_id_documents[sorted_retrieval_scores>0]
            sorted_retrieved_documents = train_corpus[sorted_id_retreved_documents]
            relevant_answers = list(valid_df[['BA1', 'BA2', 'BA3', 'BA4', 'BA5', 'BA6']].loc[valid_df.Q.values==query].values[0])
            relevant_documents = list(train_df[train_df['Utterance'].isin(relevant_answers)].Context)    
            query_precisions.append(len(set(relevant_documents) & set(sorted_retrieved_documents)) / len(set(sorted_retrieved_documents)))
            query_recalls.append(len(set(relevant_documents) & set(sorted_retrieved_documents)) / len(set(relevant_documents)))    
            query_precision_at20s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:20])) / len(set(sorted_retrieved_documents[:20])))
            query_precision_at10s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:10])) / len(set(sorted_retrieved_documents[:10])))
            query_precision_at5s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:5])) / len(set(sorted_retrieved_documents[:5])))
            query_precision_at2s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:2])) / len(set(sorted_retrieved_documents[:2])))
            query_precision_at1s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:1])) / len(set(sorted_retrieved_documents[:1])))
            # query_recall_at20s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:20])) / min(len(set(relevant_documents)), 20))
            # query_recall_at10s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:10])) / min(len(set(relevant_documents)), 10))
            # query_recall_at5s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:5])) / min(len(set(relevant_documents)), 5))
            # query_recall_at2s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:2])) / min(len(set(relevant_documents)), 2))
            query_recall_at1s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:1])) / min(len(set(relevant_documents)), 1))
            p_at_ks, rel_at_ks = [], []
            for k in range(1, 1+len(set(sorted_retrieved_documents))):
                p_at_ks.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:k])) / len(set(sorted_retrieved_documents[:k])))
                rel_at_ks.append(1 if sorted_retrieved_documents[k-1] in relevant_documents else 0)
            ave_precisions.append(sum([p*r for p, r in zip(p_at_ks, rel_at_ks)])/len(set(relevant_documents)))
            if query_recalls[-1]>0:
                for r, doc in enumerate(sorted_retrieved_documents):
                    if doc in relevant_documents:
                        rec_ranks.append(1/(1+r))
                        break
                else:
                    rec_ranks.append(0)
        check.append([query, sorted_retrieved_documents[:3], relevant_documents])
            
    print("Mean Average Precision (MAP): ", np.mean(ave_precisions))
    print("Mean Reciprocal Rank (MRR): ", np.mean(rec_ranks))
    print("Mean precision: ", np.mean(query_precisions))
    print("Mean recall: ", np.mean(query_recalls))
    print("Mean precision @20: ", np.mean(query_precision_at20s))
    print("Mean precision @10: ", np.mean(query_precision_at10s))
    print("Mean precision @5: ", np.mean(query_precision_at5s))
    print("Mean precision @2: ", np.mean(query_precision_at2s))
    print("Mean precision @1: ", np.mean(query_precision_at1s))
    # print("Mean recall @20: ", np.mean(query_recall_at20s))
    # print("Mean recall @10: ", np.mean(query_recall_at10s))
    # print("Mean recall @5: ", np.mean(query_recall_at5s))
    # print("Mean recall @2: ", np.mean(query_recall_at2s))
    print("Mean recall @1: ", np.mean(query_recall_at1s))

print_metrics(y, yhat)

k=50 #( or rank > 0.5 or thr: select only classified as 1)
pred_answers = np.take(KBanswers, list(np.argsort(rankings, axis=0)[::-1][:k]))
pred_answers = [answer[0] for answer in pred_answers]
step1_corpus = list(knowledgebase.loc[knowledgebase.Utterance.isin(pred_answers), 'Context'])
tokenized_corpus = [doc.split(" ") for doc in step1_corpus]

k=5
bm25 = BM25Okapi(tokenized_corpus)
step2_rankings = bm25.get_scores(input_text.split(" "))
step2_ranked_questions = np.take(step1_corpus, list(np.argsort(step2_rankings, axis=0)[::-1][:k]))
step2_answers = [knowledgebase.loc[knowledgebase.Context == a, "Utterance"] for a in step2_ranked_questions]

print("Reranked answers: ", step2_answers[0])
        
        

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