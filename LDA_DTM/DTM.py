import logging
import os
from gensim import corpora, utils
from gensim.models.wrappers.dtmmodel import DtmModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from gensimLDA import *  # dependency on gensimLDA.py

"""
Processes saved news articles into a Dynamic topic model
Plots chosen index terms and their probability distribution in a given topic

The input needs to be as follows:
*.txt files, encoded in the standard ACII encoding format that have been cleaned from any punctuation marks and
any other non text elements.
I did this using a simple bash script, included in this repository

"""
source_folder = home/example  # folder that contains all the news articles
dtm_compiled_path = home/example/dtm  # direct path to the C-Code compiled form David Blei's git
                                      # (https://github.com/blei-lab/dtm)
custom_stop_words = ['zeit', 'is', 'a', 'the']  # should be adapted to the source of articles and content


g = gensimLDA()
g.change_operating_language('de')
g.generate_stop_words(custom=custom_stop_words)
# I named articles per {$month}+{article number}, such as 304 for the 5th selected article in March
articles = [str(100+i)+'c' for i in range(13)] + [str(200+i)+'c' for i in range(11)]\
           +[str(300+i)+'c' for i in range(11)] + [str(400+i)+'c' for i in range(10)] \
           + [str(500+i)+'c' for i in range(15)]
g.define_filenames_and_foldername_source(articles, source_folder)
g.load_raw_corpus()
time_slices = [13, 11, 11, 10, 15]  # number of documents for each month


class DTMcorpus(corpora.textcorpus.TextCorpus):

    def get_texts(self):
        return self.input

    def __len__(self):
        return len(self.input)

corpus = DTMcorpus(g.corpus_raw)
model = DtmModel(dtm_compiled_path, corpus, time_slices, num_topics=2, id2word=corpus.dictionary, initialize_lda=True)


# collect probabilities for chosen keyterms
words_of_interest = ['Tuerkei', 'Fluechtlinge', 'Oesterreich']
topic_choice = 0
results = {}
for w in words_of_interest:
    results[w] = []
for i in range(5):
    for (p,w) in model.show_topic(topic_choice, i):
        if w in int_words:
            results[w].append(p)

# plot
labels = ["January '16", "February '16", "March '16", "April '16'", "Mai '16"]
line1, = plt.plot(results['Fluechtlinge'], label='"Refugee"', marker='o')
line2, = plt.plot(results['Oesterreich'], label='"Austria"', marker='o')
line3, = plt.plot(results['Tuerkei'], label='"Turkey"', marker='o')
plt.xticks([0, 1, 2, 3, 4], labels, rotation=70)
plt.ylabel='Word probability'
plt.xlabel='Publishing month of article'
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, bbox_to_anchor=(1.3, 1))
plt.show()

