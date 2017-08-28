import pandas as pd
import numpy as np
import loadIBC as libc
import treeUtil as tutil
import enchant
import re
import spacy

from nltk.corpus import stopwords
from string import punctuation as punc


def clean_df(path, nlp):
    '''
    INPUT: filepath, SpaCy dictionary in language of choice
    OUTPUT: cleaned Pandas DataFrame

    IMPORT THIS FUNCTION WHEN USING TO CLEAN A NEW DOCUMENT.
    '''
    df = pd.read_csv(path)
    df.columns = ['sent', 'label']
    df = _lemmatize_and_remove_stops(df, nlp)
    return df

def _lemmatize_and_remove_stops(df, nlp):
    '''
    INPUT: DF from clean_df, SpaCy dictionary in language of choice
    OUTPUT: DF cleaned to no longer include stop words, numbers, or punctuation. All words are lemmatized using SpaCy.
    '''
    df.sent = df.sent.apply(lambda words: _remove_stops(words))
    df.sent = df.sent.apply(lambda words: _lemmatize_with_spacy(words, nlp))
    return df

def _remove_stops(sent):
    '''
    INPUT: String of words
    OUTPUT: Cleaned string without stop words, digits, or punctuation.
    '''
    d = enchant.Dict('en_US')
    return re.sub("[0-9]", "", ' '.join([word.lower().strip(punc) for word in list(sent.split(' ')) if word not in set(stopwords.words('english')) and d.check(word) == True]))

def _lemmatize_with_spacy(sent, nlp):
    '''
    INPUT: string of words, SpaCy dictionary
    OUTPUT: Lemmatized string of words
    '''
    replacement, originals = nlp(sent), []
    for ix, token in enumerate(replacement):
        originals.append(token.lemma_)
    return ' '.join(originals)


if __name__ == '__main__':
    # completed initial load
    # libc.load_data()
    # completed initial cleaning
    # nlp = spacy.load('en')
    # df = clean_df('data.csv', nlp)
    # print(df.columns.tolist())
    print('Initial_cleaning completed already.')
