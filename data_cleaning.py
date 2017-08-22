import pandas as pd
import numpy as np
import loadIBC as libc
import treeUtil as tutil
import enchant
import re
import os
import spacy

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from string import punctuation as punc


def check_df(path, nlp):
    df = pd.read_csv(path)
    df.columns = ['sent', 'label']
    df = remove_stop_words(df, nlp)
    return df

def remove_stop_words(df, nlp):
    df.sent = df.sent.apply(lambda words: _check_for_stops(words))
    df.sent = df.sent.apply(lambda words: _lemmatize_with_spacy(words, nlp))
    return df

def _check_for_stops(sent):
    d = enchant.Dict('en_US')
    return re.sub("[0-9]", "", ' '.join([word.lower().strip(punc) for word in list(sent.split(' ')) if word not in set(stopwords.words('english')) and d.check(word) == True]))

def _lemmatize_with_spacy(sent, nlp):
    replacement, originals = nlp(sent), []
    for ix, token in enumerate(replacement):
        originals.append(token.lemma_)
    return ' '.join(originals)

def get_tfidf(df):
    v = tfidf()
    x = v.fit_transform(df['sent']).to_dense()
    print(type(x))
    return df

# change to_dense to make sparse matrix into np array and add into pandas df with labels - save that as tfidt_data.csv
# combine that all into one function with hidden methods


if __name__ == '__main__':
    # completed initial load
    # libc.load_data()
    # completed initial cleaning
    # nlp = spacy.load('en')
    # df = check_df('data.csv', nlp)
    # print(df.columns.tolist())
    # df.to_csv(path_or_buf='cleaned_data.csv', header=['sent', 'label'])
    print('Initial_cleaning completed already.')
