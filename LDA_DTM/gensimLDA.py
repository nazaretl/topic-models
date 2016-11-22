from stop_words import get_stop_words
from gensim import corpora,  models, similarities
import pandas as pd
"""
Expected input (as clearified in the function descriptions):
- txt files cleaned of any punctuation  marks (for example using the bash script, located in this repository)
- any saved mm corpus and dictionary produced by other gensim scripts
    - including the output produced by the wiki make script from gensim (https://radimrehurek.com/gensim/wiki.html)
"""


class gensimLDA(object):
    """ Collection of functions that work on the Python library gensim to work queries against a LDA model
     based on a document corpus"""
    def __init__(self):
        # Preprocessing
        self.operating_language = 'en'
        self.source_filenames = None
        self.source_foldername = None
        self.save_foldername = None
        self.save_prefix = None
        self.query_filename = None
        self.query_foldername = None
        # Corpora
        self.corpus_raw= None
        self.corpus = None
        self.documents = None
        self.stoplist = None
        self.token_count = None
        self.corpus_dict = None
        self.corpus_mm = None
        # Model
        self.lda = None
        # Query
        self.query_list = {}
        self.query_dict = {}
        self.query_results = {}
        self.query_results_topics = {}
        self.query_results_topics_probs = {}
        self.query_results_non_sparse = {}

    # Paths

    def define_filenames_and_foldername_source(self, filenames, foldername):
        """
        Define names of the documents that are meant to be included in the corpus
        TRAINING DATA
        """
        self.source_filenames = filenames
        self.source_foldername = foldername

    def define_filenames_and_foldername_save(self, prefix, foldername):
        """
        Define where documents shall be saved and using which prefix
        SAVED DATA
        """
        self.save_prefix = prefix
        self.save_foldername = foldername

    def define_filename_and_foldername_query(self, filename, foldername):
        """
        Define where documents shall be saved and using which prefix
        SAVED DATA
        """
        self.query_filename = filename
        self.query_foldername = foldername

    # Preprocessing

    def change_operating_language(self, lang):
        """
        Change the operating language (for stopwords list)
        Default: English ('en')
        """
        self.operating_language = lang

    # Corpus Creation

    def generate_stop_words(self, numbers='include', custom=[]):
        """
        Generate a list of stop words
        Per default: Include single numbers and no custom words
        :param: custom - include custom words (lower case)
        """
        self.stoplist = set(get_stop_words(self.operating_language))
        if numbers == 'include':
            self.stoplist.update(range(100)+ [10**i for i in range(5)])
        self.stoplist.update(custom)

    def load_raw_corpus(self):
        """
        Load from txt file and convert into list
        Verify: Document must be cleaned of any punctuation marks (,.!'";: , etc)
        """
        if len(self.stoplist) == 0:
            print('Empty stop word list')
        corpus = []
        for filename in self.source_filenames:
            temp = []
            with open(self.source_foldername+'/'+filename + '.txt', 'r') as f:
                for line in f:
                    for word in line.split():
                        if word.lower() not in self.stoplist:
                            temp.append(word.lower())
            corpus.append(temp)
        self.corpus_raw = corpus
        print(str(len(self.corpus_raw)) + ' loaded into corpus')

    def create_frequency_count(self):
        """Create a dictionary that contains all word counts"""
        dict = {}
        for document in self.corpus_raw:
            for token in document:
                try:
                    dict[token] += 1
                except KeyError:
                    dict[token] = 1
        self.token_count = dict

    def remove_once(self):
        """Remove all words that occur only once over the whole corpus"""
        self.corpus = [[token for token in document if self.token_count[token] > 1] for document in self.corpus_raw]

    def save_dict(self):
        """ Save the corpus as a gensim dictionary for future use"""
        dictionary = corpora.Dictionary(self.corpus)
        dictionary.save(self.save_foldername+'/'+self.save_prefix+'.dict')
        self.corpus_dict = dictionary

    def save_mm(self):
        """ Saves the coprus in gensim's Mm Corpus format"""
        corpus = [self.corpus_dict.doc2bow(document) for document in self.corpus]
        corpora.MmCorpus.serialize(self.save_foldername+'/'+self.save_prefix+'.mm', corpus)
        self.corpus_mm = corpus

    def load_corpus_mm(self):
        """ Load a corpus from memory - Format: *.mm"""
        corpus = corpora.MmCorpus(self.save_foldername + '/' + self.save_prefix + '_bow.mm')
        self.corpus_mm = corpus

    def load_corpus_dict(self):
        """ Load a corpus from memory - Format: *.dict"""
        corpus = corpora.Dictionary.load(self.save_foldername + '/' + self.save_prefix + '.dict')
        self.corpus_dict = corpus

    def load_corpus_dict_from_text(self):
        """ Load coprus from a gensim created dictionary .txt file"""
        id2word = corpora.Dictionary.load_from_text(self.save_foldername + '/' +
                                                           self.save_prefix + '_wordids.txt')
        self.corpus_dict = id2word

    def train_lda_model(self, topics=100, chunksize=1000, passes = 3):
        """ Runs the LDA model with k topics"""
        self.lda = models.LdaModel(self.corpus_mm, id2word=self.corpus_dict, num_topics=topics, update_every=1,
                                   chunksize=chunksize, passes=passes)
        print('LDA model trained')

    def save_lda_model(self):
        """Save the LDA model"""
        self.lda.save(self.save_foldername+'/'+self.save_prefix)

    def create_single_line_query(self, query):
        """ Add a new query - one string"""
        if len(self.stoplist) == 0:
            print('Empty stop word list')
        temp = []
        for word in query.split():
            if word.lower() not in self.stoplist:
                temp.append(word.lower())
        self.query_list = temp

    def create_new_query_from_raw(self):
        """ Add a new query from a txt file - without any punctuation marks"""
        if type(self.stoplist)!= set:
            print('Stop word list not yet generated')
        temp = []
        with open(self.query_foldername + '/' + self.query_filename + '.txt', 'r') as f:
            for line in f:
                for word in line.split():
                    if word.lower() not in self.stoplist:
                        temp.append(word.lower())
        self.query_list[self.query_filename] = temp

    def transform_query_to_dict(self):
        """Take saved query in list form and transforms it to a bow vector"""
        try:
            self.query_dict[self.query_filename] = self.corpus_dict.doc2bow(self.query_list[self.query_filename])
        except AttributeError:
            print('No corpus dict loaded!')

    def search_query(self):
        """Search the query against the LDA model"""
        self.query_results[self.query_filename] = self.lda[self.query_dict[self.query_filename]]
        return self.query_results

    def query_top_topics(self, topics):
        """Returns the k most matching topics"""
        dict = {}
        for i in self.query_results[self.query_filename]:
            dict[i[0]] = i[1]
        self.query_results_topics_probs[self.query_filename] = pd.Series(dict).sort_values(ascending=False).iloc[:topics].to_dict()
        self.query_results_topics[self.query_filename] = self.query_results_topics_probs[self.query_filename].keys()

