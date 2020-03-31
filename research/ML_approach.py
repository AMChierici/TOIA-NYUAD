#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 07:31:27 2019

@author: amc
"""

import pandas as pd
import numpy as np
import collections
import re
import torch
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
# Load model
from InferSent.encoder.models import InferSent
import matplotlib.pyplot as plt
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
from sklearn.linear_model import LogisticRegression, Lasso
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

   
zeroturndialogue = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/MargaritaCorpusKB.csv', encoding='utf-8')
multiturndialogues = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/DIALOGUES.csv', encoding='utf-8')


def allnull(somelist):
    count=0
    for i in somelist:
        if pd.isnull(i):
            count+=1
    return count==len(somelist)

#BIG dataset
#Context, WOzAnswers, Labels= [], [], []
#for example_id in range(len(multiturndialogues)):
#    exampleWOzAnswers = list(multiturndialogues.iloc[example_id, 3:].values)
#    tmp_WAs = []
#    allAs = list(zeroturndialogue.Answer.values)
#    if allnull(exampleWOzAnswers):
#        Context.append(multiturndialogues.iloc[example_id, 1])
#        WOzAnswers.append("I cannot answer this question")
#        Labels.append(1)
#    else:  
#        for distr in allAs:
#            if distr in exampleWOzAnswers:
#                Context.append(multiturndialogues.iloc[example_id, 1])
#                WOzAnswers.append(distr)
#                Labels.append(1)
#            else:
#                Context.append(multiturndialogues.iloc[example_id, 1])
#                WOzAnswers.append(distr)
#                Labels.append(0)
                
                
## alternative
Context, WOzAnswers, Labels= [], [], []
allAs = list(zeroturndialogue.Utterance.values)
for example_id in range(len(multiturndialogues)):
    exampleWOzAnswers = list(multiturndialogues.iloc[example_id, 7:].values)
    no_distr_to_sample = 0
    ids_to_exclude = []
    if allnull(exampleWOzAnswers):
        Context.append(multiturndialogues.iloc[example_id, 4])
        WOzAnswers.append("I don't have an answer for this question")
        Labels.append(1)
        Context.append(multiturndialogues.iloc[example_id, 4])
        np.random.seed(example_id)
        WOzAnswers.append(str(np.random.choice(allAs, 1)[0]))
        Labels.append(0)
    else:  
        for answer in exampleWOzAnswers:
            if not pd.isnull(answer):
                Context.append(multiturndialogues.iloc[example_id, 4])
                WOzAnswers.append(answer)
                Labels.append(1)
                ids_to_exclude.append(allAs.index(answer))
                no_distr_to_sample += 1      
        #
        tmp_distractors = [allAs[i] for i in
                np.array(range(len(allAs)))
                [np.isin(range(len(allAs)), ids_to_exclude, invert=True)]
                ]
        #
        np.random.seed(example_id)
        if no_distr_to_sample==1:
            np.random.seed(example_id)
            answer = str(np.random.choice(tmp_distractors, 10))
        else:    
            for answer in np.random.choice(tmp_distractors, no_distr_to_sample*10, replace=False):
                Context.append(multiturndialogues.iloc[example_id, 4])
                WOzAnswers.append(str(answer))
                Labels.append(0)

 
               
data_df = pd.DataFrame({'Context':Context, 'Utterance':WOzAnswers, 'Label':Labels})
tmp_df = pd.DataFrame({'Context':zeroturndialogue.Context.values, 'Utterance':zeroturndialogue.Utterance.values, 'Label':[1 for i in range(len(zeroturndialogue))]})
data_df = data_df.append(tmp_df)
data_df.reset_index(level=None, drop=True, inplace=True)
               

train_df, test_df = train_test_split(data_df, test_size=0.25, random_state=45)
test_df, valid_df = train_test_split(test_df, test_size=0.5, random_state=45)

########## USING InferSent ##########
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
for item in train_df.Utterance.values: allsentences.append(item)
#for item in zeroturndialogue.Answer.values: allsentences.append(item)
model.build_vocab(allsentences, tokenize=True)


context_embeddings = model.encode(train_df.Context.values, bsize=128, tokenize=False, verbose=True)
answer_embeddings = model.encode(train_df.Utterance.values, bsize=128, tokenize=False, verbose=True)
################################


########## USING BERT ##########
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
    
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

#test = bertembed("hi, how are you?")

### Opt 1
context_embeddings = []
for text in train_df.Context.values:
    context_embeddings.append(bertembed(text)[0])
context_embeddings = np.array(context_embeddings)
answer_embeddings = []
for text in train_df.Utterance.values:
    answer_embeddings.append(bertembed(text)[0])
answer_embeddings = np.array(answer_embeddings)
X_train = context_embeddings - answer_embeddings
### Opt2
# alternative concatenating QA pair (I guess more accurate for what BERT is):
X_train = []
for question, answer in zip(train_df.Context.values, train_df.Utterance.values):
    text = question + " [SEP] " + answer
    X_train.append(bertembed(text)[0])
X_train = np.array(X_train)
###
y_train = train_df.Label.values


regr_ln = LogisticRegression(penalty="l2")
regr_ln.fit(X_train, y_train)

# Predict on train data
y_ln = regr_ln.predict(X_train)
y_ln_probs = regr_ln.predict_proba(X_train)
TN, FP, FN, TP = confusion_matrix(y_train, y_ln, labels=[1,0]).ravel()
TN, FP, FN, TP
accuracy_score(y_train, y_ln)
f1_score(y_train, y_ln)
recall_score(y_train, y_ln, labels=[1,0])
precision_score(y_train, y_ln, labels=[1,0])
precision_recall = precision_recall_curve(y_train, y_ln_probs[:,1])
plt.scatter(precision_recall[0], precision_recall[1], color='blue')

### Opt1
context_embeddings = []
for text in valid_df.Context.values:
    context_embeddings.append(bertembed(text)[0])
context_embeddings = np.array(context_embeddings)
answer_embeddings = []
for text in valid_df.Utterance.values:
    answer_embeddings.append(bertembed(text)[0])
answer_embeddings = np.array(answer_embeddings)
X_valid = context_embeddings - answer_embeddings
### Opt2
X_valid = []
for question, answer in zip(valid_df.Context.values, valid_df.Utterance.values):
    text = question + " [SEP] " + answer
    X_valid.append(bertembed(text)[0])
X_valid = np.array(X_valid)
###
y_valid = valid_df.Label.values

y_ln = regr_ln.predict(X_valid)
y_ln_probs = regr_ln.predict_proba(X_valid)
TN, FP, FN, TP = confusion_matrix(y_valid, y_ln, labels=[1,0]).ravel()
TN, FP, FN, TP
accuracy_score(y_valid, y_ln)
f1_score(y_valid, y_ln)
recall_score(y_valid, y_ln, labels=[1,0])
precision_score(y_valid, y_ln, labels=[1,0])
precision_recall = precision_recall_curve(y_valid, y_ln_probs[:,1])
plt.scatter(precision_recall[0], precision_recall[1], color='blue')

thr = TP/(2*TP + FN + FP)
#optimizing decision thr: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4442797/

### Opt1
context_embeddings = []
for text in test_df.Context.values:
    context_embeddings.append(bertembed(text)[0])
context_embeddings = np.array(context_embeddings)
answer_embeddings = []
for text in test_df.Utterance.values:
    answer_embeddings.append(bertembed(text)[0])
answer_embeddings = np.array(answer_embeddings)
X_test = context_embeddings - answer_embeddings
### Opt2
X_test = []
for question, answer in zip(test_df.Context.values, test_df.Utterance.values):
    text = question + " [SEP] " + answer
    X_test.append(bertembed(text)[0])
X_test = np.array(X_test)
###
y_test = test_df.Label.values

y_ln_probs = regr_ln.predict_proba(X_test)
precision_recall = precision_recall_curve(y_test, y_ln_probs[:,1])
plt.scatter(precision_recall[0], precision_recall[1], color='blue')

y_ln = (regr_ln.predict_proba(X_test)[:,1] >= thr).astype(bool)
print("\n CM: ", confusion_matrix(y_test, y_ln, labels=[1,0]),
    "\n Accuracy: ", accuracy_score(y_test, y_ln),
    "\n Precision: ", precision_score(y_test, y_ln),
    "\n Recall: ", recall_score(y_test, y_ln),
    "\n F1: ", f1_score(y_test, y_ln),
)

###btw, I am not using cosine similarity at all here


# Predict on test data
##InferSent
context_embeddings = model.encode(test_df.Context.values, bsize=128, tokenize=False, verbose=True)
answer_embeddings = model.encode(test_df.Utterance.values, bsize=128, tokenize=False, verbose=True)
##BERT
context_embeddings = []
for text in test_df.Context.values:
    context_embeddings.append(bertembed(text)[0])
context_embeddings = np.array(context_embeddings)
answer_embeddings = []
for text in test_df.Utterance.values:
    answer_embeddings.append(bertembed(text)[0])
answer_embeddings = np.array(answer_embeddings)

X_test = context_embeddings - answer_embeddings

X_test = []
for question, answer in zip(test_df.Context.values, test_df.Utterance.values):
    text = question + " [SEP] " + answer
    X_test.append(bertembed(text)[0])
X_test = np.array(X_test)

y_test = test_df.Label.values
y_ln = regr_ln.predict(X_test)
y_ln_probs = regr_ln.predict_proba(X_test)
cm = confusion_matrix(y_test, y_ln, labels=[1,0])
accuracy_score(y_test, y_ln)
f1_score(y_test, y_ln)
recall_score(y_test, y_ln, labels=[1,0])
precision_recall = precision_recall_curve(y_test, y_ln_probs[:,1])
precision_score(y_test, y_ln, labels=[1,0])
plt.scatter(precision_recall[0], precision_recall[1], color='blue')


thr = TP/(2*TP + FN + FP)
#optimizing decision thr: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4442797/
y_ln = (regr_ln.predict_proba(X_test)[:,1] >= thr).astype(bool)
print(confusion_matrix(y_test, y_ln, labels=[1,0]),
    accuracy_score(y_test, y_ln),
    f1_score(y_test, y_ln),
    recall_score(y_test, y_ln),
    precision_score(y_test, y_ln),
)

## Random Forest
class_rf = RandomForestClassifier(random_state=45, max_features='auto', n_estimators= 200, max_depth=8, criterion='gini')
class_rf.fit(X_train, y_train)
y_rf = class_rf.predict(X_train)
y_rf_probs = class_rf.predict_proba(X_train)
confusion_matrix(y_train, y_rf, labels=[1,0])
accuracy_score(y_train, y_rf)
recall_score(y_train, y_rf, labels=[1,0])
precision_score(y_train, y_rf, labels=[1,0])
f1_score(y_train, y_rf, labels=[1,0])
precision_recall = precision_recall_curve(y_train, y_rf_probs[:,1])
plt.scatter(precision_recall[0], precision_recall[1], color='blue')

y_rf = class_rf.predict(X_test)
y_rf_probs = class_rf.predict_proba(X_test)
confusion_matrix(y_test, y_rf, labels=[1,0])
accuracy_score(y_test, y_rf)
recall_score(y_test, y_rf, labels=[1,0])
precision_score(y_test, y_rf, labels=[1,0])
f1_score(y_test, y_rf, labels=[1,0])
precision_recall = precision_recall_curve(y_test, y_rf_probs[:,1])
plt.scatter(precision_recall[0], precision_recall[1], color='blue')





answers = list(np.unique(allAs)) + list(["I don't have an answer for this question"])
#distractors = list(zeroturndialogue.Answer.values) + list(["I cannot answer this question"])
#context_embeddings = model.encode(test_df.Context.values, bsize=128, tokenize=False, verbose=True)
#answer_embeddings = model.encode(list(zeroturndialogue.Answer.values) + list(["I cannot answer this question"]), bsize=128, tokenize=False, verbose=True)

input_text = "Where are you from?"

predictions = []
for A in answers:
    text = input_text + " [SEP] " + A
    predictions.append(bertembed(text)[0])
    
k=10

#ex_id=15
#X_test = [context_embeddings[ex_id] - A for A in answers]

rankings = regr_ln.predict_proba(predictions)

print("Question: ", input_text)
for answer, ranking in zip(np.take(answers, list(np.argsort(rankings[:,1], axis=0)[::-1][:k])), np.sort(rankings[:,1], axis=0)[::-1][:k]):
    print(
          "\n Answer: ", answer,
          "\n [Rank value: ", math.floor(ranking*1000), "]"
      )
    
predicted = (regr_ln.predict_proba(predictions)[:,1] >= thr).astype(bool)
from collections import Counter
Counter(predicted)
[print(x) for x, y in zip(answers, predicted) if y]

#It looksl like the ranking results are all the same. Perhaps becasue many are "I cannot answer..." Though the value of the prediction changes. They are the most frequent answers, mening that model just sees too many examples where those are the answers. Perhaps then remove the too common ones and model separately?

test_df.Context.values[ex_id]
test_df.Utterance.values[ex_id]
test_df.Label.values[ex_id]
pre_id=46
distractors[pre_id]

train_df.Utterance.value_counts()[0:10]

  ##### end of not-sure  


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

###< courtesy of https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74 >###
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=45, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


rf_random.best_params_
rf_random.best_score_
rf_random.best_estimator_
cvrf = rf_random.best_estimator_

cvrf.fit(X_train, y_train)

#y_crf = cvrf.predict(X_test)
#y_crf_probs = cvrf.predict_proba(X_test)
#confusion_matrix(y_test, y_crf, labels=[1,0])
#accuracy_score(y_test, y_crf)  
#recall_score(y_test, y_crf, labels=[1,0])
#precision_score(y_test, y_crf, labels=[1,0])
#f1_score(y_test, y_crf, labels=[1,0])
#precision_recall = precision_recall_curve(y_test, y_crf_probs[:,1])
#plt.scatter(precision_recall[0], precision_recall[1], color='blue')

y_crf = cvrf.predict(X_valid)
y_crf_probs = cvrf.predict_proba(X_valid)
TN, FP, FN, TP = confusion_matrix(y_valid, y_crf, labels=[1,0]).ravel()
TN, FP, FN, TP
accuracy_score(y_valid, y_crf)
f1_score(y_valid, y_crf)
recall_score(y_valid, y_crf, labels=[1,0])
precision_score(y_valid, y_crf, labels=[1,0])
precision_recall = precision_recall_curve(y_valid, y_crf_probs[:,1])
plt.scatter(precision_recall[0], precision_recall[1], color='blue')

thr = TP/(2*TP + FN + FP)
#optimizing decision thr: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4442797/

y_crf_probs = cvrf.predict_proba(X_test)
precision_recall = precision_recall_curve(y_test, y_crf_probs[:,1])
plt.scatter(precision_recall[0], precision_recall[1], color='blue')

y_crf = (cvrf.predict_proba(X_test)[:,1] >= thr).astype(bool)
print("\n CM: ", confusion_matrix(y_test, y_crf, labels=[1,0]),
    "\n Accuracy: ", accuracy_score(y_test, y_crf),
    "\n Precision: ", precision_score(y_test, y_crf),
    "\n Recall: ", recall_score(y_test, y_crf),
    "\n F1: ", f1_score(y_test, y_crf),
)


input_text = "Tell me about your family"

predictions = []
for A in answers:
    text = input_text + " [SEP] " + A
    predictions.append(bertembed(text)[0])
    
k=10
rankings = cvrf.predict_proba(predictions)

print("Question: ", input_text)
for answer, ranking in zip(np.take(answers, list(np.argsort(rankings[:,1], axis=0)[::-1][:k])), np.sort(rankings[:,1], axis=0)[::-1][:k]):
    print(
          "\n Answer: ", answer,
          "\n [Rank value: ", math.floor(ranking*1000), "]"
      )
    
predicted = (cvrf.predict_proba(predictions)[:,1] >= thr).astype(bool)
from collections import Counter
Counter(predicted)
[print(x) for x, y in zip(answers, predicted) if y]


#
#
#def evaluate(model, test_features, test_labels):
#    predictions = model.predict(test_features)
#    f1 = f1_score(test_labels, predictions, labels=[1,0])
#    print('Model Performance')
#    print('f1 = {:0.4f}%.'.format(100*f1))  
#    return f1
#
#base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
#base_model.fit(X_train, y_train)
#base_f1 = evaluate(base_model, X_train, y_train)
#
#best_random = rf_random.best_estimator_
#random_f1 = evaluate(best_random, X_test, y_test)
#
#
#

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

import matplotlib.pyplot as plt
plt.style.use("ggplot")
clf = RandomForestClassifier(n_jobs=-1)

param_grid = {
    'min_samples_split': [2, 10], 
    'n_estimators' : [100, 500, 1000, 2000],
    'max_depth': [10, 50, 100],
    'max_features': ['auto', 'sqrt']
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}

def grid_search_wrapper(refit_score='precision_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=3)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # make the predictions
    y_pred = grid_search.predict(X_valid)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the validation data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_valid, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=[0, 1]))
    return grid_search

grid_search_clf = grid_search_wrapper(refit_score='recall_score')

results = pd.DataFrame(grid_search_clf.cv_results_)
results = results.sort_values(by='mean_test_recall_score', ascending=False)
print(results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head())

y_scores = grid_search_clf.predict_proba(X_test)[:, 1]
# for classifiers with decision_function, this achieves similar results
# y_scores = classifier.decision_function(X_test)

p, r, thresholds = precision_recall_curve(y_test, y_scores)

def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

def precision_recall_threshold(p, r, thresholds, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    
    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0.5, 1.01]);
    plt.xlim([0.5, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)
    
    
 precision_recall_threshold(p, r, thresholds, 0.35)   
 
 def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    f1s = 2*(recalls*precisions)/(recalls+precisions)
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.plot(thresholds, f1s[:-1], "d-", label="F1 Score")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    
# use the same p, r, thresholds that were previously calculated
plot_precision_recall_vs_threshold(p, r, thresholds)

def plot_roc_curve(fpr, tpr, label=None):
    """
    The ROC curve, modified from 
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')
    
    
fpr, tpr, auc_thresholds = roc_curve(y_test, y_scores)
print(auc(fpr, tpr)) # AUC of ROC
plot_roc_curve(fpr, tpr, 'recall_optimized')



y_clf = grid_search_clf.predict(X_valid)
y_clf_probs = grid_search_clf.predict_proba(X_valid)
TN, FP, FN, TP = confusion_matrix(y_valid, y_clf, labels=[1,0]).ravel()
TN, FP, FN, TP
thr = TP/(2*TP + FN + FP)
y_clf = (grid_search_clf.predict_proba(X_test)[:,1] >= .35).astype(bool)
print("\n CM: ", confusion_matrix(y_test, y_clf, labels=[1,0]),
    "\n Accuracy: ", accuracy_score(y_test, y_clf),
    "\n Precision: ", precision_score(y_test, y_clf),
    "\n Recall: ", recall_score(y_test, y_clf),
    "\n F1: ", f1_score(y_test, y_clf),
)




import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

nnet = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
# from tensorflow.keras.optimizers import RMSprop
nnet.compile(optimizer=tf.optimizers.Adam(), #RMSprop(lr=0.001)
              loss='binary_crossentropy',
              metrics=['accuracy'])

nnet.fit(X_train, y_train, epochs=20, callbacks=[callbacks])

y_nnet_probs = nnet.predict(X_valid, batch_size=10)
y_nnet = [1 if i>.5 else 0 for i in y_nnet_probs]
TN, FP, FN, TP = confusion_matrix(y_valid, y_nnet, labels=[1,0]).ravel()
TN, FP, FN, TP
accuracy_score(y_valid, y_nnet)
f1_score(y_valid, y_nnet)
recall_score(y_valid, y_nnet, labels=[1,0])
precision_score(y_valid, y_nnet, labels=[1,0])
thr = TP/(2*TP + FN + FP)

### Opt1
context_embeddings = []
for text in test_df.Context.values:
    context_embeddings.append(bertembed(text)[0])
context_embeddings = np.array(context_embeddings)
answer_embeddings = []
for text in test_df.Utterance.values:
    answer_embeddings.append(bertembed(text)[0])
answer_embeddings = np.array(answer_embeddings)
X_test = context_embeddings - answer_embeddings
### Opt2
X_test = []
for question, answer in zip(test_df.Context.values, test_df.Utterance.values):
    text = question + " [SEP] " + answer
    X_test.append(bertembed(text)[0])
X_test = np.array(X_test)
###
y_test = test_df.Label.values

y_nnet_probs = nnet.predict(X_test, batch_size=10)
precision_recall = precision_recall_curve(y_test, y_nnet_probs)
plt.scatter(precision_recall[0], precision_recall[1], color='blue')

y_nnet = [1 if i>thr else 0 for i in y_nnet_probs]
print("\n CM: ", confusion_matrix(y_test, y_nnet, labels=[1,0]),
    "\n Accuracy: ", accuracy_score(y_test, y_nnet),
    "\n Precision: ", precision_score(y_test, y_nnet),
    "\n Recall: ", recall_score(y_test, y_nnet),
    "\n F1: ", f1_score(y_test, y_nnet),
)


input_text = "How are the students dorms?"

predictions = []
for A in answer_embeddings:
    predictions.append(bertembed(input_text)[0]-A)
predictions = np.vstack(predictions)
    
k=10
rankings = nnet.predict(predictions)

print("Question: ", input_text)
for answer, ranking in zip(np.take(test_df.Utterance.values, list(np.argsort(rankings, axis=0)[::-1][:k])), np.sort(rankings, axis=0)[::-1][:k]):
    print(
          "\n Answer: ", answer,
          "\n [Rank value: ", math.floor(ranking*1000), "]"
      )
    
