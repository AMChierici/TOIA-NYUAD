#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:44:45 2020

@author: amc
"""

import numpy as np

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


# for each query:
    
query_precisions, query_recalls, query_precision_at20s, query_recall_at20s, query_precision_at10s, query_recall_at10s,  query_precision_at5s, query_recall_at5s,  query_precision_at2s, query_recall_at2s,  query_precision_at1s, query_recall_at1s, ave_precisions = [], [], [], [], [], [], [], [], [], [], [], [], []
for query, retrieval_scores in zip(list(valid_df.Q.values), y):
    # documents=corpus
    sorted_retrieval_scores = np.sort(retrieval_scores, axis=0)[::-1]
    if sorted_retrieval_scores[0]==0:
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
    else:
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
        query_recall_at20s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:20])) / len(set(relevant_documents)))
        query_recall_at10s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:10])) / len(set(relevant_documents)))
        query_recall_at5s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:5])) / len(set(relevant_documents)))
        query_recall_at2s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:2])) / len(set(relevant_documents)))
        query_recall_at1s.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:1])) / len(set(relevant_documents)))
        p_at_ks, rel_at_ks = [], []
        for k in range(1, 1+len(set(sorted_retrieved_documents))):
            p_at_ks.append(len(set(relevant_documents) & set(sorted_retrieved_documents[:k])) / len(set(sorted_retrieved_documents[:k])))
            rel_at_ks.append(1 if sorted_retrieved_documents[k-1] in relevant_documents else 0)
        ave_precisions.append(sum([p*r for p, r in zip(p_at_ks, rel_at_ks)])/len(set(relevant_documents)))
        
        
print("Mean Average Precision (MAP): ", np.mean(ave_precisions))
print("Mean precision: ", np.mean(query_precisions))
print("Mean recall: ", np.mean(query_recalls))
print("Mean precision @20: ", np.mean(query_precision_at20s))
print("Mean precision @10: ", np.mean(query_precision_at10s))
print("Mean precision @5: ", np.mean(query_precision_at5s))
print("Mean precision @2: ", np.mean(query_precision_at2s))
print("Mean precision @1: ", np.mean(query_precision_at1s))
print("Mean recall @20: ", np.mean(query_recall_at20s))
print("Mean recall @10: ", np.mean(query_recall_at10s))
print("Mean recall @5: ", np.mean(query_recall_at5s))
print("Mean recall @2: ", np.mean(query_recall_at2s))
print("Mean recall @1: ", np.mean(query_recall_at1s))


y = [pred.predict(test_df.Q[x], list(train_corpus)) for x in range(len(valid_df))]