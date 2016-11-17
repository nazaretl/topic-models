from gensim import models, corpora
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class VisualTFIDF(object):
    """Visualizes a TF-IDF matrix as created through the gensim Python extension"""
    def __init__(self):
        self.tfidf = None
        self.tfidf_as_pd = None
        self.tfidf_as_pd_labeled = None
        self.id2word = None

    def load_tfidf(self, path):
        """Load a preprocessed saved tfidf matrix"""
        self.tfidf = models.TfidfModel.load(path)

    def load_dict(self, path):
        self.id2word = corpora.Dictionary.load_from_text(path)

    def make_pandas_vec(self):
        """ Make a pandas vector"""
        tf = self.tfidf.dfs
        idf = self.tfidf.idfs
        p = pd.concat([pd.Series(tf), pd.Series(idf)], axis=1)
        p.columns = ['df', 'idf']
        self.tfidf_as_pd = p

    def pandas_vec_add_labels(self):
        """Replace the id with the corresponding token"""
        ind = []
        for i in range(len(self.tfidf_as_pd)):
            ind.append(self.id2word[i])
        self.tfidf_as_pd_labeled = self.tfidf_as_pd
        self.tfidf_as_pd_labeled.index = pd.Series(ind)

    def barplot_term_frequencies(self, top):
        """Create a bar plot with descending term frequency for the top k terms"""
        od = np.arange(top)
        ordered_pd = self.tfidf_as_pd_labeled.sort_values('df', ascending=False)
        plt.bar(od, ordered_pd['df'][:top], align='center', alpha=0.4)
        # plt.xticks(od, ordered_pd.index[:top])
        plt.xticks([int(0.1 * top * i) for i in range(int(0.01 * top))],
                   [self.tfidf_as_pd_labeled.sort_values('df', ascending=False).index[int(0.1 * top * i)] for i in range(int(0.01 * top))], rotation=70)
        plt.ylabel('Number of documents')
        plt.show()
