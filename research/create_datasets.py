#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 5 17:19:59 2019

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
nltk.download('stopwords')
ps = SnowballStemmer('english')
   
zeroturndialogue = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/ooc_data.csv', encoding='ISO-8859-1')
multiturndialogues = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/newconversations_woz.csv', encoding='ISO-8859-1')
multiturndialogues_no_ooc_used = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/newconversations_woz_onlynewdata.csv', encoding='ISO-8859-1')

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


############# Test Data for simple, one turn, using ooc as training baselione ############
    #here, we use as test only new data questions that have an answer in the ooc data. If no answer, we disregard (though it might be useful to use anyway those questions w/out answers to check what the tf-idf would rank first. Perhaps we find other sensible answers!)
def test_set_baseline_ooctrain(multiturndialogues):
    Context = []
    WOzAnswers = []
    Distractors = []
    
    for example_id in range(len(multiturndialogues)):
        exampleWOzAnswers = list(multiturndialogues.iloc[example_id, 3:].values)
        #if not allnull(exampleWOzAnswers):
        tmp_WAs = []
        tmp_distrs = []
        for distr in list(zeroturndialogue.Answer.values):
            if distr in exampleWOzAnswers:
                tmp_WAs.append(distr)
            else:
                tmp_distrs.append(distr)
        Context.append(multiturndialogues.iloc[example_id, 1])
        WOzAnswers.append(tmp_WAs)
        Distractors.append(tmp_distrs)

    return pd.DataFrame({'Context':Context, 'WOzAnswers':WOzAnswers, 'Distractors':Distractors})        
 ############# Test Data for simple, one turn, using ooc as training baselione ############       
        


#for i in range(len(dialogue1)):
#    dialogue1.iloc[i, 0] = preprocess(dialogue1.iloc[i, 0])
#    dialogue1.iloc[i, 1] = preprocess(dialogue1.iloc[i, 1])
#    

#sample context windows - a window is a number of turns, defined as 1qs and 1ans
#actually, make the window from full dialogue to min 3plets: 1qs, 1a, 1qs - thene we can try random window size, or just one size sliding, e.g. 3, 4, 5, ...etc.
# To start with, I pick 2 2plets: Q-A-Q-A-Q-A-Q +A

def add_eos(list_of_docs):
    return ' <eos> '.join(list_of_docs)
#try
#add_eos(list(multiturndialogues.Q.values[:3] + " <eos> " + multiturndialogues.A.values[:3]))

def train_test_sets_seqdata(TURNS = 1, hold_out_sample = np.array([3, 7])):
    #THIS FUNCTION NEEDS WORK: the exampleWOzAnswers include the ground truth utterance but this is not into the zeroturndialogue, so all indeces 0 in y_test will never be in the exampleWOzAnswers. Here we are expanding the range of QAs so we shall append train data QA pairs to the ooc QA pairs
    
    conversation_IDs = np.unique(multiturndialogues.ConversationID)
    train_samples = conversation_IDs[np.isin(conversation_IDs, hold_out_sample, invert=True)]
    test_samples = conversation_IDs[np.isin(conversation_IDs, hold_out_sample)]
    
    
    if TURNS==0:
        train_df = multiturndialogues.loc[multiturndialogues.ConversationID.isin(train_samples)][['Q','A']]
        train_df.reset_index(level=None, drop=True, inplace=True)
        train_df = train_df.rename(columns = {"Q":"Context", "A": "Utterance"})
        test_df = multiturndialogues.loc[multiturndialogues.ConversationID.isin(test_samples)]
        test_df.reset_index(level=None, drop=True, inplace=True)
        Context = []
        WOzAnswers = []
        Distractors = []
        for example_id in range(len(test_df)):
            exampleWOzAnswers = list(test_df.iloc[example_id, 2:].values)
    #        if not allnull(exampleWOzAnswers):
            tmp_WAs = []
            tmp_distrs = []
            for distr in np.unique(zeroturndialogue.Answer.values):
                if distr in exampleWOzAnswers:
                    tmp_WAs.append(distr)
                else:
                    tmp_distrs.append(distr)
            Context.append(test_df.iloc[example_id, 1])
            WOzAnswers.append(tmp_WAs)
            Distractors.append(tmp_distrs)
        
    
    if TURNS>0:
        #train_df
        train_df = pd.DataFrame({'Context':[], 'Utterance':[]})
        
        for dial in train_samples:
            #re-write this step: here we should just select a list of sequential Q-A according to par TURNS then combine them with the add_eos fn.
            tmp = multiturndialogues.loc[multiturndialogues.ConversationID==dial]['Q'] + ' <eos> ' + multiturndialogues.loc[multiturndialogues.ConversationID==dial]['A']
            tmp.reset_index(level=None, drop=True, inplace=True)
            Q, A = [], []
            for i in range(len(tmp)-TURNS-1):
                Q.append(add_eos(tmp.loc[i:(i+(TURNS-1))]))
                Q[-1] += ' <eos> ' + tmp.loc[i+TURNS].split(' <eos> ')[0]
                A.append(tmp.loc[i+(TURNS)].split(' <eos> ')[1])
            df = pd.DataFrame({'Context':Q, 'Utterance':A})
            train_df = train_df.append(df, ignore_index = True) 
    
        
        #test_df      
        for dial in test_samples:
            tmp = multiturndialogues.loc[multiturndialogues.ConversationID==dial]['Q'] + ' <eos> ' + multiturndialogues.loc[multiturndialogues.ConversationID==dial]['A']
            tmp.reset_index(level=None, drop=True, inplace=True)
            tmp_WozAnswers = multiturndialogues.loc[multiturndialogues.ConversationID==dial]
            tmp_WozAnswers.reset_index(level=None, drop=True, inplace=True)
            
            Context, WOzAnswers, Distractors = [], [], []
            for i in range(len(tmp)-TURNS-1):
                Context.append(add_eos(tmp.loc[i:(i+(TURNS-1))]))
                Context[-1] += ' <eos> ' + tmp.loc[i+TURNS].split(' <eos> ')[0]
                A.append(tmp.loc[i+TURNS].split(' <eos> ')[1])
                ##################
                exampleWOzAnswers = list(tmp_WozAnswers.iloc[i+TURNS, 2:].values)
        #        if not allnull(exampleWOzAnswers):
                tmp_WAs = []
                tmp_distrs = []
                for distr in np.unique(zeroturndialogue.Answer.values):
                    if distr in exampleWOzAnswers:
                        tmp_WAs.append(distr)
                    else:
                        tmp_distrs.append(distr)
                WOzAnswers.append(tmp_WAs)
                Distractors.append(tmp_distrs)
                
    test_df = pd.DataFrame({'Context':Context, 'WOzAnswers':WOzAnswers, 'Distractors':Distractors})        
            
        
    return train_df, test_df
    

# Data Preparation

def build_vocabularies(words, n_words):
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



# combined two lists in one list of tubles
def listOfTuples(l1, l2): 
	return list(map(lambda x, y:(x,y), l1, l2)) 


def evaluate_recall_modified(y, y_test, k=1):
    # modifying this to allow more labels to be checked if in the preditions or not. Although this is quite simplistic now. We could counte how many correct ones are
    num_examples = float(len(y))
    num_correct = 0
    for predictions, labels in zip(y, y_test):
        intersection = set(labels) & set(predictions[:k])
        if intersection:
            num_correct += len(list(intersection))
    return num_correct/num_examples

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

#ind=13
#predictions = np.argsort(y[ind], axis=0)[::-1]
#sorted_scores = np.sort(y[ind], axis=0)[::-1]
#A,B = set(y_valid[ind]), set([item for item, value in zip(predictions[:20], sorted_scores[:20]) if value>0.7])
#intersection = A & B
#
#valid_df.iloc[ind, 0]
#(valid_df.iloc[ind, 1]+valid_df.iloc[ind, 2])[170]



def predict_random(context, utterances, N):
    return np.random.choice(len(utterances), N, replace=False)
    
    
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



def test_set_questions_ooctrain(multiturndialogues):
    # modified to use index of answers in test WOzAnswers --> replaced with WOzAnswersIDs
    Context, WOzAnswersIDs = [], []
    
    for example_id in range(len(multiturndialogues)):
        exampleWOzAnswers = list(multiturndialogues.iloc[example_id, 3:].values)
#        if not allnull(exampleWOzAnswers):
        tmp_WAs = []
        allAs = list(zeroturndialogue.Answer.values)
        for distr in allAs:
            if distr in exampleWOzAnswers:
                tmp_WAs.append(allAs.index(distr))
        Context.append(multiturndialogues.iloc[example_id, 1])
        WOzAnswersIDs.append(tmp_WAs)

    
    return pd.DataFrame({'Context':Context, 'WOzAnswers':WOzAnswersIDs})    



def train_test_sets_questions_seqdata_no_ooc(TURNS = 1, hold_out_sample = np.array([3,7]), valid_sample = np.array([8,9])):
#    hold_out_sample = np.array([1, 2]) HAD TO CHANGE THIS because when building the test set, the WOz "did not see" the answer from other test set conversations, so the test set can only be 1 conversation. And even, this way it's tricky as the WoZ recycled in some cases the answers from the same conversation.
    
    conversation_IDs = np.unique(multiturndialogues_no_ooc_used.ConversationID)
    train_samples = conversation_IDs[np.isin(conversation_IDs, np.concatenate((hold_out_sample, valid_sample)), invert=True)]
    valid_samples = conversation_IDs[np.isin(conversation_IDs, valid_sample)]
    test_samples = conversation_IDs[np.isin(conversation_IDs, hold_out_sample)]
    
    
    if TURNS==0:
        #TRAIN
        train_df = multiturndialogues_no_ooc_used.loc[multiturndialogues_no_ooc_used.ConversationID.isin(train_samples)][['Q','A']]
        train_df.reset_index(level=None, drop=True, inplace=True)
        train_df = train_df.rename(columns = {"Q":"Context", "A": "Utterance"})
        #TEST
        test_df = multiturndialogues_no_ooc_used.loc[multiturndialogues_no_ooc_used.ConversationID.isin(test_samples)]
        test_df.reset_index(level=None, drop=True, inplace=True)
        Context, WOzAnswersIDs = [], []
        allAs = list(train_df.Utterance.values)
        for example_id in range(len(test_df)):
            exampleWOzAnswers = list(test_df.iloc[example_id, 3:].values)
#            if not allnull(exampleWOzAnswers):
            tmp_WAs = []
            for distr in allAs:
                if distr in exampleWOzAnswers:
                    tmp_WAs.append(allAs.index(distr))
            Context.append(test_df.iloc[example_id, 1])
            WOzAnswersIDs.append(tmp_WAs)
        #VALIDATION
        valid_df = multiturndialogues_no_ooc_used.loc[multiturndialogues_no_ooc_used.ConversationID.isin(valid_samples)]
        valid_df.reset_index(level=None, drop=True, inplace=True)
        v_Context, v_WOzAnswersIDs = [], []
        allAs = list(train_df.Utterance.values)
        for example_id in range(len(valid_df)):
            exampleWOzAnswers = list(valid_df.iloc[example_id, 3:].values)
#            if not allnull(exampleWOzAnswers):
            tmp_WAs = []
            for distr in allAs:
                if distr in exampleWOzAnswers:
                    tmp_WAs.append(allAs.index(distr))
            v_Context.append(valid_df.iloc[example_id, 1])
            v_WOzAnswersIDs.append(tmp_WAs)
    
    
    if TURNS>0:
        #TRAIN
        train_df = pd.DataFrame({'Context':[], 'Utterance':[]})
        
        for dial in train_samples:
            #re-write this step: here we should just select a list of sequential Q-A according to par TURNS then combine them with the add_eos fn.
            tmp = multiturndialogues_no_ooc_used.loc[multiturndialogues_no_ooc_used.ConversationID==dial]['Q'] + ' <eos> ' + multiturndialogues_no_ooc_used.loc[multiturndialogues_no_ooc_used.ConversationID==dial]['A']
            tmp.reset_index(level=None, drop=True, inplace=True)
            Q, A = [], []
            for i in range(len(tmp)-TURNS-1):
                Q.append(add_eos(tmp.loc[i:(i+(TURNS-1))]))
                Q[-1] += ' <eos> ' + tmp.loc[i+TURNS].split(' <eos> ')[0]
                A.append(tmp.loc[i+(TURNS)].split(' <eos> ')[1])
            df = pd.DataFrame({'Context':Q, 'Utterance':A})
            train_df = train_df.append(df, ignore_index = True)             
        #TEST      
        for dial in test_samples:
            tmp = multiturndialogues_no_ooc_used.loc[multiturndialogues_no_ooc_used.ConversationID==dial]['Q'] + ' <eos> ' + multiturndialogues_no_ooc_used.loc[multiturndialogues_no_ooc_used.ConversationID==dial]['A']
            tmp.reset_index(level=None, drop=True, inplace=True)
            tmp_WozAnswers = multiturndialogues_no_ooc_used.loc[multiturndialogues_no_ooc_used.ConversationID==dial]
            tmp_WozAnswers.reset_index(level=None, drop=True, inplace=True)
            
            Context, WOzAnswersIDs = [], []
            allAs = list(train_df.Utterance.values)
            for i in range(len(tmp)-TURNS-1):
                Context.append(add_eos(tmp.loc[i:(i+(TURNS-1))]))
                Context[-1] += ' <eos> ' + tmp.loc[i+TURNS].split(' <eos> ')[0]
                ##################
                exampleWOzAnswers = list(tmp_WozAnswers.iloc[i+TURNS, 3:].values)
#                if not allnull(exampleWOzAnswers):
                tmp_WAs = []
                for distr in allAs:
                    if distr in exampleWOzAnswers:
                        tmp_WAs.append(allAs.index(distr))
                WOzAnswersIDs.append(tmp_WAs)
        #VALIDATION      
        for dial in valid_samples:
            tmp = multiturndialogues_no_ooc_used.loc[multiturndialogues_no_ooc_used.ConversationID==dial]['Q'] + ' <eos> ' + multiturndialogues_no_ooc_used.loc[multiturndialogues_no_ooc_used.ConversationID==dial]['A']
            tmp.reset_index(level=None, drop=True, inplace=True)
            tmp_WozAnswers = multiturndialogues_no_ooc_used.loc[multiturndialogues_no_ooc_used.ConversationID==dial]
            tmp_WozAnswers.reset_index(level=None, drop=True, inplace=True)
            
            v_Context, v_WOzAnswersIDs = [], []
            allAs = list(train_df.Utterance.values)
            for i in range(len(tmp)-TURNS-1):
                v_Context.append(add_eos(tmp.loc[i:(i+(TURNS-1))]))
                v_Context[-1] += ' <eos> ' + tmp.loc[i+TURNS].split(' <eos> ')[0]
                ##################
                exampleWOzAnswers = list(tmp_WozAnswers.iloc[i+TURNS, 3:].values)
#                if not allnull(exampleWOzAnswers):
                tmp_WAs = []
                for distr in allAs:
                    if distr in exampleWOzAnswers:
                        tmp_WAs.append(allAs.index(distr))
                v_WOzAnswersIDs.append(tmp_WAs)
            
                
    test_df = pd.DataFrame({'Context':Context, 'WOzAnswers':WOzAnswersIDs})        
    valid_df = pd.DataFrame({'Context':v_Context, 'WOzAnswers':v_WOzAnswersIDs})        
            
        
    return train_df, valid_df, test_df