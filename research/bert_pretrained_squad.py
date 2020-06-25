#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:11:48 2020

@author: amc
"""

import torch
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question = "What else do you like to do?"
answer_text = "What else do you like to do?	I love drawing sometimes, but abstract shapes, basically doodling. I love volleyball and biking a lot and I am very interested in negotiation as a science and the aspects of communication."

# Apply the tokenizer to the input text, treating them as a text-pair.
input_ids = tokenizer.encode(question, answer_text)

print('The input has a total of {:} tokens.'.format(len(input_ids)))



# BERT only needs the token IDs, but for the purpose of inspecting the 
# tokenizer's behavior, let's also get the token strings and display them.
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# For each token and its id...
for token, id in zip(tokens, input_ids):
    
    # If this is the [SEP] token, add some space around it to make it stand out.
    if id == tokenizer.sep_token_id:
        print('')
    
    # Print the token string and its ID in two columns.
    print('{:<12} {:>6,}'.format(token, id))

    if id == tokenizer.sep_token_id:
        print('')
        
# Search the input_ids for the first instance of the `[SEP]` token.
sep_index = input_ids.index(tokenizer.sep_token_id)

# The number of segment A tokens includes the [SEP] token istelf.
num_seg_a = sep_index + 1

# The remainder are segment B.
num_seg_b = len(input_ids) - num_seg_a

# Construct the list of 0s and 1s.
segment_ids = [0]*num_seg_a + [1]*num_seg_b

# There should be a segment_id for every input token.
assert len(segment_ids) == len(input_ids)

# Run our example through the model.
start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                 token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

# Find the tokens with the highest `start` and `end` scores.
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

# Combine the tokens in the answer and print it out.
answer = ' '.join(tokens[answer_start:answer_end+1])

print('Answer: "' + answer + '"')


# Start with the first token.
answer = tokens[answer_start]

# Select the remaining answer tokens and join them with whitespace.
for i in range(answer_start + 1, answer_end + 1):
    
    # If it's a subword token, then recombine it with the previous token.
    if tokens[i][0:2] == '##':
        answer += tokens[i][2:]
    
    # Otherwise, add a space then the token.
    else:
        answer += ' ' + tokens[i]

print('Answer: "' + answer + '"')

import matplotlib.pyplot as plt
import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
#sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (16,8)

# Pull the scores out of PyTorch Tensors and convert them to 1D numpy arrays.
s_scores = start_scores.detach().numpy().flatten()
e_scores = end_scores.detach().numpy().flatten()

# We'll use the tokens as the x-axis labels. In order to do that, they all need
# to be unique, so we'll add the token index to the end of each one.
token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token, i))

# Create a barplot showing the start word score for all of the tokens.
ax = sns.barplot(x=token_labels, y=s_scores, ci=None)

# Turn the xlabels vertical.
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

# Turn on the vertical grid to help align words to scores.
ax.grid(True)

plt.title('Start Word Scores')

plt.show()

import pandas as pd

# Store the tokens and scores in a DataFrame. 
# Each token will have two rows, one for its start score and one for its end
# score. The "marker" column will differentiate them. A little wacky, I know.
scores = []
for (i, token_label) in enumerate(token_labels):

    # Add the token's start score as one row.
    scores.append({'token_label': token_label, 
                   'score': s_scores[i],
                   'marker': 'start'})
    
    # Add  the token's end score as another row.
    scores.append({'token_label': token_label, 
                   'score': e_scores[i],
                   'marker': 'end'})
    
df = pd.DataFrame(scores)

# Draw a grouped barplot to show start and end scores for each word.
# The "hue" parameter is where we tell it which datapoints belong to which
# of the two series.
g = sns.catplot(x="token_label", y="score", hue="marker", data=df,
                kind="bar", height=6, aspect=4)

# Turn the xlabels vertical.
g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="center")

# Turn on the vertical grid to help align words to scores.
g.ax.grid(True)
    

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    # print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    # print('Answer: "' + answer + '"')
            
    score = float(torch.max(start_scores)) + float(torch.max(end_scores))
    
    return answer, score
    
    
import textwrap

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=80) 

dialogue_segment= "Please dance! Okay! (Dances) Dance for me! Okay! (Dances) Can you dance for me?	Okay! (Dances) Please play something! Okay! (Plays ukulele) Play something for me! Okay! (Plays ukulele) Can you play an instrument for me?	Okay! (Plays ukulele) Please sing! Okay! (Sings) Sing for me! Okay! (Sings) Can you sing for me? Okay! (Sings) OK	Tell me what else you'd like to know. uh huh	Would you like to know some more? Do you accept applicants who already hold a bachelor of arts or science degrees? At NYUAD there are only programs to get the first bachelor. When will I hear if I get admitted? Check on the university's web page! How did you prepare to apply there? I googled a lot of examples of university applications. I made a list of all my achievements and everything I want to mention. And I just asked a lot of people to proofread and give me advice. And you got in, when? I was accepted to NYUAD in December. When should I apply for NYU Abu Dhabi? I'd recommend you apply in October, a year before the university's academic year. Was it hard to apply? It was a very long application and then you had to come here for a candidate weekend and interview and then you would find out if you got in. Am I able to change campus location after I enroll? It's difficult, but you could. Also, keep in mind that scholarships are different for each campus. Does NYUAD care about diversity? NYUAD cares about all types of diversity of its students, including of their backgrounds and diversity of thought. Is the TOEFL required for consideration for admission to NYU Abu Dhabi? NYUAD does not currently require an English language proficiency test. Your English skills will be obvious in the application and later on in the interview. Did you apply to other universities, too? NYUAD was my first choice and I didn't even apply to other universities! But that's because I applied very early so I had plenty of time to figure out other universities in case I wasn't admitted."

print(wrapper.fill(dialogue_segment))

question = "What's your name?"

search_input, _ = answer_question(question, dialogue_segment)

from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np

train_df = pd.read_csv('/Users/amc/Documents/TOIA-NYUAD/research/MargaritaCorpusKB.csv', encoding='utf-8')

train_corpus = np.unique(train_df.Utterance.values)

tokenized_corpus = [doc.split(" ") for doc in train_corpus]
bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(search_input.split(" "))

i=0
for a, s in zip(
        train_corpus[np.argsort(scores, axis=0)[::-1]][:5],
        np.sort(scores, axis=0)[::-1][:5]):
    print("\n Ranked #{} Answer: {} \n --Score: {}".format(i+1, a, s))
    i+=1
    
    
train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)

question = "What's your name?"
window_size = 10
search_phrases = []
search_scores = []
for window in range(0, len(train_df), window_size):
    dialogue_segment = ' '.join(list(
        train_df.loc[window:(window + window_size), 'Context'].map(str) + ' ' + 
        train_df.loc[window:(window + window_size), 'Utterance'].map(str)
        ))
    answer, score = answer_question(question, dialogue_segment)
    search_phrases.append(answer)
    search_scores.append(score)

[search_phrases[x] for x in np.argsort(search_scores, axis=0)[::-1][:5]]
np.sort(search_scores, axis=0)[::-1][:5]

search_input = question + ' ' + search_phrases[np.argsort(search_scores, axis=0)[::-1][0]]

scores = bm25.get_scores(search_input.split(" "))
i=0
for a, s in zip(
        train_corpus[np.argsort(scores, axis=0)[::-1]][:5],
        np.sort(scores, axis=0)[::-1][:5]):
    print(wrapper.fill("Ranked #{} Answer: {} --Score: {}".format(i+1, a, s)))
    i+=1
    
# y = [bm25.get_scores(valid_df.Q[x].split(" ")) for x in range(len(valid_df))]

