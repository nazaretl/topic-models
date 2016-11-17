from gensimLDA import *
"""
Match each article with their closest (topic-distribution-wise) article
Articles need to be individual *.txt files, in standard ASCII encoding standard, cleaned of any punctuation marks and non-textual elements
I cleaned articles with a simple bash script, included in this repository
All articles are searched for by default in a folder called query/articles

Returns a dictionary with the corresponding match
"""

l = gensimLDA()
l.lda = g.lda
l.corpus_dict = g.corpus_dict
l.generate_stop_words()

# saved all query articles in a folder called query

all_query_articles = {0: 'article1-1', 1: 'article1-2', 2: 'article2-1', 3: 'article2-2',
                      4: 'article3-1', 5: 'article3-2', 6: 'article4-1', 7: 'article4-2',
                      8: 'article5-1', 9: 'article5-2', 10: 'article6-1', 11: 'article6-2',
                      12: 'article7-1', 13: 'article7-2'}

# filter meta_topics and stop word-like topics
filtered_topics = [13]

for article in all_query_articles.values():
    l.define_filename_and_foldername_query(foldername='query/articles', filename=article)
    l.create_new_query_from_raw()
    l.transform_query_to_dict()
    l.search_query()
    l.query_top_topics(5)


def dist(a,b):
    """ distance function - here: cosine similarity"""
    return spatial.distance.cosine(non_sparse[a], non_sparse[b])


def transform_to_non_sparse_vector():
    """"create non sparse vectors, same dimensions as the topic space"""
    non_sparse = {}
    for art in l.query_results:
        vec = [0 for i in range(100)]
        for (k,v) in l.query_results[art]:
            if k not in filtered_topics:
                vec[k] = v
        non_sparse[art] = vec
    return non_sparse

non_sparse = transform_to_non_sparse_vector()


def create_matches():
    """ Creates a matrix that matches documents based on min distance"""
    matrix = np.zeros((len(non_sparse),len(non_sparse)))
    for i in range(len(non_sparse)):
        for j in range(len(non_sparse)):
            if i==j:
                matrix[i,j] = 1.0
            else:
                matrix[i,j] = dist(all_query_articles[i], all_query_articles[j])
    matrix[matrix==0]=1
    match = {}
    for i in all_query_articles:
        match[all_query_articles[i]] = all_query_articles[matrix[i].argmin()]
    return match

match = create_matches()
