#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:39:10 2020
Ref: https://plotly.com/python/creating-and-updating-figures/
@author: amc
"""
import pandas as pd
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go

kb_file = '/Users/amc/Documents/TOIA-NYUAD/research/MargaritaCorpusKB.csv'
dialogues_file = '/Users/amc/Documents/TOIA-NYUAD/research/DIALOGUES.csv'

knowledgebase = pd.read_csv(kb_file, encoding='utf-8')
train_test_dialogues = pd.read_csv(dialogues_file, encoding='utf-8')

validation_dialogues = train_test_dialogues.loc[
    train_test_dialogues.Experiment == "TRAIN"]
test_dialogues = train_test_dialogues.loc[
    train_test_dialogues.Experiment == "TEST"]


lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')


def first_trigram(sentence):
    """

    Parameters
    ----------
    sentence : text
        sentence you want to extract first trigram from.

    Returns
    -------
    tuple
        first trigram, lowered, punctuation removed, lemmatized.

    """
    sentence_tokenized = tokenizer.tokenize(sentence)
    sentence_lemmatized = [lemmatizer.lemmatize(t.lower())
                           for t in sentence_tokenized]
    trigrams = list(ngrams(sentence_lemmatized, 3, pad_right=True))
    return trigrams[0]


def df_express_sunburst(data, text_column):
    """

    Parameters
    ----------
    data : Pandas DataFrame
        Data frame with original text column.
    text_column : text
        name of text column.

    Returns
    -------
    Pandas Dataframe
        table of data ready for plotting sunburst chart.

    """
    trigram_list = [first_trigram(question)
                    for question in
                    pd.unique(data[data[text_column] != '*']
                              [text_column])]
    df_kb = pd.DataFrame.from_dict(Counter(trigram_list),
                                   orient='index',
                                   columns=['count'])
    df_kb = df_kb.reset_index()
    df_kb['1st_word'] = pd.Series([trigram[0]
                                   for trigram in df_kb['index']])
    df_kb['2nd_word'] = pd.Series([trigram[1]
                                   for trigram in df_kb['index']])
    df_kb['3rd_word'] = pd.Series([trigram[2]
                                   for trigram in df_kb['index']])
    df_kb.rename(columns={"index": "first_trigram"})
    df_kb.replace(to_replace=[None], value="", inplace=True)
    return df_kb


def df_go_sunburst(df):
    """

    Parameters
    ----------
    df : Pandas DataFrame
        pd df formatted by prep_sunburst_df.

    Returns
    -------
    Pandas DataFrame
        pd df formatted for plotting with go.

    """
    # We make sure df has index reset to avoid problems when stacking
    df = df.reset_index(drop=True)
    # Start by grouping count by first unigram and save df_tmp
    df_tmp = df.groupby('1st_word').sum('count').reset_index()
    ids = pd.Series(['total - {}'.format(word)
                     for word in df_tmp['1st_word'].tolist()])
    parents = pd.Series(['total']*df_tmp.shape[0])
    df_tmp['parents'] = parents
    df_tmp['ids'] = ids
    df_tmp = df_tmp.rename(columns={"1st_word": "labels"})

    # Then grorup count by first bigram and save in df_tmp
    df_tmp2 = df.copy()
    bigrams = pd.Series([i + ' - ' + j if j != "" else i
                           for i, j in zip(df_tmp2['1st_word'],
                                           df_tmp2['2nd_word'])])
    df_tmp2['bigrams'] = bigrams
    # reset index without drop, gives us the indeces in a column
    df_tmp2 = df_tmp2.groupby('bigrams').sum('count').reset_index()
    ids = pd.Series(['total - {}'.format(word)
                     for word in df_tmp2['bigrams']])
    parents = pd.Series(['total' + ' - ' + bigram.split(' - ')[0] if
                         len(bigram.split(" - ")) > 1 else 'total'
                         for bigram in df_tmp2['bigrams']])
    labels = pd.Series(
        [bigram.split(' - ')[1] if len(bigram.split(" - ")) > 1 else bigram
                         for bigram in df_tmp2['bigrams']])
    df_tmp2['parents'] = parents
    df_tmp2['ids'] = ids
    df_tmp2['labels'] = labels
    df_tmp2 = df_tmp2.drop(columns=['bigrams'])
    # Make sure we don't repeat ids - they must be unique
    df_tmp2 = df_tmp2[~(df_tmp2['ids'].
                        isin(df_tmp['ids']))].reset_index(drop=True)

    # Then grorup count by first trigram and save in df_tmp3. Here is
    # mostly renaming.
    df_tmp3 = df.copy()
    ids = pd.Series(['total - {} - {} - {}'.
                     format(trigram[0], trigram[1], trigram[2]) if
                     ((trigram[1] != None) and (trigram[2] != None)) else
                     'total - {} - {}'.format(trigram[0], trigram[1]) if
                     trigram[1] != None else 'total - {}'.format(trigram[0])
                     for trigram in df_tmp3['index']])
    parents = pd.Series(['total - {} - {}'.format(trigram[0], trigram[1]) if
                         trigram[1] != None else 'total - {}'.
                         format(trigram[0]) for trigram in df_tmp3['index']])
    df_tmp3['ids'] = ids
    df_tmp3['parents'] = parents
    df_tmp3 = df_tmp3.rename(columns={"3rd_word": "labels"})
    df_tmp3 = df_tmp3.drop(columns=['index', '1st_word', '2nd_word'])
    # Make sure we don't repeat ids - they must be unique.
    # Check both in tmp2 ...
    df_tmp3 = df_tmp3[~(df_tmp3['ids'].
                        isin(df_tmp2['ids']))].reset_index(drop=True)
    # ... And in tmp
    df_tmp3 = df_tmp3[~(df_tmp3['ids'].
                        isin(df_tmp['ids']))].reset_index(drop=True)

    # Add "total" as the root
    df_total = pd.DataFrame({
        "parents": [""],
        "labels": ['total'],
        "ids": ['total'],
        "count": [df['count'].sum()]})

    # Finally stack all of them together
    df_output = df_total.append([df_tmp, df_tmp2, df_tmp3],
                                    ignore_index=True)
    df_output = df_output.rename(columns={"count": "values"})
    # Should be redundant, but better be safe to remove duplicate rows
    df_output = df_output.drop_duplicates()
    # Again should be redundant, but remove leaves that end empty
    df_output = df_output[df_output['labels'] != ""]
    return df_output.reset_index(drop=True)


df_kb = df_express_sunburst(knowledgebase, "Context")
df_kb_sub = df_kb[df_kb['count'] > 2]
fig = px.sunburst(df_kb_sub, path=['1st_word', '2nd_word', '3rd_word'],
                  values='count')
plot(fig, filename='train_xp.html')

df_valid_dial = df_express_sunburst(validation_dialogues, "Q")
df_valid_dial_sub = df_valid_dial[df_valid_dial['count'] > 2]
fig = px.sunburst(df_valid_dial_sub, path=['1st_word', '2nd_word', '3rd_word'],
                  values='count')
plot(fig, filename='dev_xp.html')

df_test_dial = df_express_sunburst(test_dialogues, "Q")
df_test_dial_sub = df_test_dial[df_test_dial['count'] > 2]
fig = px.sunburst(df_test_dial_sub, path=['1st_word', '2nd_word', '3rd_word'],
                  values='count')
plot(fig, filename='test_xp.html')

df_answers = df_express_sunburst(knowledgebase, "Utterance")
df_answers_sub = df_answers[df_answers['count'] > 2]
fig = px.sunburst(df_answers_sub, path=['1st_word', '2nd_word', '3rd_word'],
                  values='count')
plot(fig, filename='train_answers_xp.html')

# Now we plot for finer subset, i.e., trigrams with more than 1 occurrence.
df_kb_sub = df_go_sunburst(df_kb[df_kb['count'] > 1])
fig = go.Figure(go.Sunburst(
    ids=df_kb_sub['ids'].tolist(),
    labels=df_kb_sub['labels'].tolist(),
    parents=df_kb_sub['parents'].tolist(),
    values=df_kb_sub['values'].tolist()
    ))
# Update layout for tight margin
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
plot(fig, filename='train.html')

df_valid_dial_sub = df_go_sunburst(df_valid_dial[df_valid_dial['count'] > 1])
fig = go.Figure(go.Sunburst(
    ids=df_valid_dial_sub['ids'].tolist(),
    labels=df_valid_dial_sub['labels'].tolist(),
    parents=df_valid_dial_sub['parents'].tolist(),
    values=df_valid_dial_sub['values'].tolist()
    ))
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
plot(fig, filename='dev.html')

df_test_dial_sub = df_go_sunburst(df_test_dial[df_test_dial['count'] > 1])
fig = go.Figure(go.Sunburst(
    ids=df_test_dial_sub['ids'].tolist(),
    labels=df_test_dial_sub['labels'].tolist(),
    parents=df_test_dial_sub['parents'].tolist(),
    values=df_test_dial_sub['values'].tolist()
    ))
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
plot(fig, filename='test.html')

df_answers_sub = df_go_sunburst(df_answers[df_answers['count'] > 1])
fig = go.Figure(go.Sunburst(
    ids=df_answers_sub['ids'].tolist(),
    labels=df_answers_sub['labels'].tolist(),
    parents=df_answers_sub['parents'].tolist(),
    values=df_answers_sub['values'].tolist()
    ))
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
plot(fig, filename='train_answers.html')
