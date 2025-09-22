from __future__ import absolute_import

from gensim.models import Word2Vec, word2vec
from nltk import word_tokenize
from gensim.models import KeyedVectors
from utils.StopWords import read_EN_stopwords, remove_stopwords
from pathConfig import project_dir
from gensim import models
import os

w2v_model_fpath=os.path.join(project_dir,"src/_1_question_retrieval/_2_word2vec_model/model/SO_vectors_200.bin")
vocab_fpath=os.path.join(project_dir,"src/_1_question_retrieval/_3_IDF_vocabulary/idf_vocab.csv")

def load_w2v_model():
    word2vector_model = KeyedVectors.load_word2vec_format(w2v_model_fpath, binary=True)
    return word2vector_model


def load_idf_vocab():
    import csv
    vocab_dict = dict()
    try:
        with open(vocab_fpath, 'r',encoding="utf-8") as csvfile:
            rd = csv.reader(csvfile)
            next(rd)
            for row in rd:
                vocab_dict[str(row[0])] = float(row[1])
    except Exception as e:
        print(f"UnicodeDecodeError: {e}")
    return vocab_dict


def preprocessing_for_query(q):
    # basic preprocessing for query
    qw = word_tokenize(q.lower())
    stopwords = read_EN_stopwords()
    qw = remove_stopwords(qw, stopwords)
    return qw
