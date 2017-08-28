import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf

def get_tfidf(df):
    df.label = df.label.apply(lambda x: 1 if x == 'lib' \
        else 0 if x == 'con' else -1)
    v = tfidf()
    return pd.DataFrame(v.fit_transform(df['sent']).toarray())

# change to_dense to make sparse matrix into np array and add into pandas df with labels - save that as tfidt_data.csv
# combine that all into one function with hidden methods


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids.

        INPUT: Xtr, the transformed tfidf matrix '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

if __name__ == '__main__':
    sparse_matrix = get_tfidf(pd.read_csv('cleaned_data.csv', index_col=0))
    print('got sparse')
