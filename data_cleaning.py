import pandas as pd
import numpy as np
import loadIBC as libc
import treeUtil as tutil
import enchant
import re
import os
import spacy

from nltk.corpus import stopwords


def check_df(path, nlp):
    df = pd.read_csv(path)
    df.columns = ['sent', 'type']
    df = remove_stop_words(df, nlp)
    return df

def remove_stop_words(df, nlp):
    df.sent = df.sent.apply(lambda word_list: _check_for_stops(word_list))
    df.sent = df.sent.apply(lambda sentence: lemmatize_with_spacy(sentence, nlp))
    return df

def _check_for_stops(sent):
    d = enchant.Dict('en_US')
    return re.sub("[0-9]", "", ' '.join([word for word in list(sent.split(' ')) if word not in set(stopwords.words('english')) and d.check(word) == True]))

def lemmatize_with_spacy(sent, nlp):
    sent1 = sent.split(' ')
    sent2 = nlp(sent)
    for ix, token in enumerate(sent2):
        sent1[ix] = token.lemma_
    return ' '.join(sent1)

if __name__ == '__main__':
    #libc.load_data()
    nlp = spacy.load('en')
    df = check_df('data.csv', nlp)
    print(df.head(2))
