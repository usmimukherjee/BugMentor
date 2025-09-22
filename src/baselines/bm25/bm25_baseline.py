import pandas as pd
import re
import string
import os
import warnings
from rank_bm25 import BM25Okapi
from nltk import word_tokenize
import numpy as np
import json
warnings.filterwarnings("ignore")

def tokenizing_corpus(corpus):
    tokens_from_corpus = []
    for s in corpus:
        cleanedTex = re.sub(r'[^\w\s]', '', str(s)).lower()
        words = (word_tokenize(cleanedTex))
        tokens_from_corpus.append(words)
    # print(tokens_generated)
    return tokens_from_corpus

def tokenizing_query(sentence):
    tokens_from_query = []
    cleanedTex=re.sub(r'[^\w\s]','',str(sentence)).lower()
    words = (word_tokenize(cleanedTex))
    tokens_from_query.append(words)
    tokens_from_query = tokens_from_query[0]
    # print(tokens_generated)
    return tokens_from_query

def calculating_bm25_scores(query,corpus):
    #arguments - query,corpus
    #return np array
    # print("\n scores:")
    tokens_from_corpus = tokenizing_corpus(corpus)
    bm25 = BM25Okapi(tokens_from_corpus)
    tokens_from_queries = tokenizing_query(query)
    docs = bm25.get_top_n(tokens_from_queries,corpus,n=10)
    return docs

def applying_on_dataset(filename):
    global gold
    global corpus
    gold = pd.read_csv("../../../data/bugmentor_gold/"+filename)
    gold.IssueID = gold.IssueID.astype('str')
    gold['Title+Description'] = gold['Title'] + gold['Description']
    gold['Title+Description+Question'] = gold['Title'] + gold['Description'] + gold['Question']

    corpus = pd.read_csv("../../../data/2__bug_reports_comments/"+filename,encoding='iso-8859-1',dtype=str, low_memory=False)
    corpus = corpus.rename(columns={'Body': 'Description'})
    corpus['Title+Description'] = corpus['Title'] + corpus['Description']
    corpus['Title+Description+Question'] = corpus['Title'] + corpus['Description'] + corpus['Question']
    corpus.IssueID = corpus.IssueID.astype('str')
    issueidslist = list(corpus.IssueID)
    # results_dataframe = pd.DataFrame(columns=['IssueID','Title','Description','Question','Result','GroundTruth'])
    # iterating over each row of gold dataframe
    for index, row in gold.iterrows():
        query = row['Question']
        results = calculating_bm25_scores(query,corpus['Comment'])
        ground_truth = row['RelevantAnswerDetail']
        # save the results and ground truth in a csv file as a dataframe
        results_df = pd.DataFrame(results, columns=['Result'])
        results_df['IssueID'] = row.IssueID
        results_df['Title'] = row.Title
        results_df['Description'] = row.Description
        results_df['Question'] = row.Question
        results_df['GroundTruth'] = ground_truth
        base_filename, extension = os.path.splitext(filename)
        name_path = "../../../results/baselines/bm25/"+base_filename+'/'+str(row.IssueID)+'.csv'
        results_df.to_csv(name_path, index=False)

if __name__ == '__main__':
    filelist = os.listdir("../../../data/bugmentor_gold/")
    filelist_sorted = sorted(filelist, reverse=True)
    for name in filelist_sorted:
        print("working on --------------"+name+"----------------------")
        base_filename, extension = os.path.splitext(name)
        directory_path = "../../../results/baselines/bm25/"+base_filename
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
        applying_on_dataset(name)
        print("done with --------------"+name+"----------------------")
        # break