import pandas as pd
import numpy as np
import loadIBC as libc
import treeUtil as tutil
import enchant

from nltk.corpus import stopwords


def check_df(path):
    df = pd.read_csv(path)
    df.columns = ['sent', 'type']
    df = remove_stop_words(df)
    return df

def remove_stop_words(df):
    df.sent = df.sent.apply(lambda word_list: _check_for_stops(word_list))
    return df

def _check_for_stops(sent):
    d = enchant.Dict('en_US')
    sent = [word for word in list(sent.split(' ')) if word not in set(stopwords.words('english')) and d.check(word) == True and word.isdigit() == False]
    sent = ' '.join(sent)
    return sent


if __name__ == '__main__':
    libc.load_data()
    df = check_df('data.csv')
    print(df.head(2))
