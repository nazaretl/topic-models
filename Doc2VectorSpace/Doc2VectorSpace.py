import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

"""
Represents large documents with at least partially overlapping vocabulary in a 3d vector space

Input is expected to be a txt file, cleaned of any punctuation marks and in ASCII encoding format
I did this using a simple bash script, included in this repository
"""


# File names
names = ['joyce', 'austen', 'tolstoi']
# Choose 3 index terms
ax_wo = ['love', 'God', 'time']
# Colors in order
clrs = ['m', 'g', 'b', 'r', 'y']


def file_to_corpus(names):
    """
    Read in txt files with ending *.txt, return corpus
    files have to be cleaned of any punctuations marks, especially quotes
    files need to be stored in the same folder as this python code
    """
    corpus = []
    for n in names:
        temp = []
        with open(n+'.txt', 'r') as f:
            for line in f:
                for word in line.split():
                    temp.append(word)
        corpus.append(temp)
    return corpus


def corpus_to_dataframe(corp):
    """Insert coprus, receive Pandas data frame"""
    return pd.DataFrame(corp).transpose()


def create_vocabulary_distinct_from_corp(corp_pd):
    """ Insert corpus as data frame, receive distinct vocabulary"""
    pds = pd.Series()
    for d in corp_pd.columns:
        pds = pds.append(corp_pd[d])
    return pds.unique()


def corpus_to_vecs(corp_pd,vocab):
    """Insert corpus as data frame, receive corpus documents as vectors in a dictionary"""
    con = pd.DataFrame()
    for c in corp_pd.columns:
        con = pd.concat([con, corp_pd[c].value_counts()], axis=1)
    con = con.fillna(0)   # replace nan values with 0
    return con


def ex():
    """Executes the functions in order"""
    corpus = file_to_corpus(names)
    cdf = corpus_to_dataframe(corpus)
    vocab = create_vocabulary_distinct_from_corp(cdf)
    cc = corpus_to_vecs(cdf, vocab)
    return cc


cc = ex()
cct = cc.transpose()
# A normalization of the vectors length can be added by dividing each column by the length of the document

def visualise():
    """ Make a 3d plot with the given vectors"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(ax_wo[0])
    ax.set_ylabel(ax_wo[1])
    ax.set_zlabel(ax_wo[2])
    ax.set_xlim([0, 800])  # adapt to length of vectors for better visibility
    ax.set_ylim([0, 700])  #
    ax.set_zlim([0, 1700])  #
    c = 0
    for i in range(len(cct)):
        plt.plot([0, cct[ax_wo[0]][i]], [0, cct[ax_wo[1]][i]], [0, cct[ax_wo[2]][i]], 'x'+clrs[c]+'-')
        # change color
        if c < len(clrs):
            c += 1
        else:
            c = 0
    return plt.show()

visualise()