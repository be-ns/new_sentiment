try:
   import cPickle as pickle
except:
   import pickle
import csv


def load_data():
    f = open('data.csv', 'w')
    w = csv.writer(f, delimiter = ',')

    [lib, con, neutral] = pickle.load(open('ibcData.pkl', 'rb'))

    for tree in lib:
        w.writerow([tree.get_words(), 'lib'])
    for tree in con:
        w.writerow([tree.get_words(), 'con'])
    for tree in neutral:
        w.writerow([tree.get_words(), 'neut'])

    f.close()

    # see treeUtil.py for the tree class definition
    # for node in ex_tree:
    #
    #     # remember, only certain nodes have labels (see paper for details)
    #     if hasattr(node, 'label'):
    #         print(node.label, ': ', node.get_words())

if __name__ == '__main__':
    load_data()
