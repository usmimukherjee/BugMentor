# imports
from datetime import datetime
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from elasticsearch.helpers import bulk
from tabulate import tabulate
import json
import warnings
warnings.filterwarnings("ignore")

# global variables
global doc
global doc_scores

# indexing the corpus
def index_the_corpus(corpus,columnname):
    corpus = corpus[['IssueID',columnname]]

    bulk_data = []
    for i, row in corpus.iterrows():
        bulk_data.append(
            {
                "_index": "bug_report",
                "_id": i,
                "_source": {
                    "IssueID": row["IssueID"],
                    columnname: row[columnname],
                }
            }
        )
    bulk(es, bulk_data, raise_on_error=False)
    es.indices.refresh(index="bug_report")
    es.cat.count(index="bug_report", format="json")

def Merge(dict1, dict2):
    return(dict2.update(dict1))

def calculating_bm25_scores(query):
    
    print("issue list : ",issueidslist)
    sorterIndex = dict(zip(issueidslist, range(len(issueidslist))))

    # querying in elastic search
    resp = es.search(index="bug_report", query={"query_string": {"query": str(query)}}, size=len(corpus))

    doc_scores = []
    issues = []

    for hit in resp['hits']['hits']:
        issues.append(hit['_source']['IssueID'])
        doc_scores.append(hit['_score'])  # returns score

    # checking if the hit issue list is a part of the corpus issue list
    check1 = set(issues).issubset(issueidslist)
    # assert (check1 == True)
    print("checking if the hit issue list is a part of the corpus issue list",check1)

    # getting the issues that isn't in the hit list
    set1 = set(issueidslist)
    set2 = set(issues)
    missing = set(set1).difference(set2)
    print("missing: ", missing)

    # checking if the hit issues are in the missing list
    check2 = set(issues).issubset(missing)

    assert(check2 == False)

    if (check1 != True and check2 != False):
        file1 = open("error.txt", "a")  # append mode
        file1.write(name)
        file1.close()

    print("checking if the hit issues are in the missing list",check2)

    # creating a dictionary of issue ids and scores for both missing issue ids with score -1000 and hit list with their
    # respective scores
    res1 = {}
    for key in missing:
        res1[key] = -1
    print("the missing issue ids: ",len(res1))

    res2 = {}
    for key in issues:
        for value in doc_scores:
            res2[key] = value
    print("the hit issue ids: ",len(res2))

    # merging both dictionaries
    Merge(res1,res2)
    print("Resultant dictionary is : " + str(res2))
    print(len(res2))

    # converting dictionary to series
    s = pd.Series(res2, name='Scores')
    s.index.name = 'IssueIds'

    # converting to a dataframe with issueid and scores as columns
    df = pd.DataFrame({'IssueIds': s.index, 'Scores': s.values})

    # mappping the order of issues with the current order of issue list with their respective scores
    df['IssueIds_Rank'] = df['IssueIds'].map(sorterIndex)
    df = df.sort_values('IssueIds_Rank')

    print("len of issue list : ", len(issueidslist))
    assert(len(issueidslist) == len(np.array(df.Scores)))
    print("len of the array of the bm25 score",len(np.array(df.Scores)))


    check3 = len(issueidslist) == len(np.array(df.Scores))

    if (check1 != True and check2 != False and check3!=True):
        print(name)
        file1 = open("error.txt", "a")  # append mode
        file1.write(name,"\n")
        file1.close()

    return np.array(df.Scores)

def ca_wise_relevance_scores(bug_example, relevance_score_df, issue):
    relevance_scores_ca_wise = {}

    # index ca1 here
    columnname = 'Candidate_Answer_1'
    index_the_corpus(corpus,columnname)
    # scores for candidate answers corpus ca1
    score_ti_ca1i = calculating_bm25_scores(bug_example['Title'], )
    score_di_ca1i = calculating_bm25_scores(bug_example['Description'])
    score_bi_ca1i = calculating_bm25_scores(bug_example['Title+Description'])
    score_qi_ca1i = calculating_bm25_scores(bug_example['Question'])
    score_tdqi_ca1i = calculating_bm25_scores(bug_example['Title+Description+Question'])

    # index ca2 here
    columnname = 'Candidate_Answer_2'
    index_the_corpus(corpus,columnname)
    # scores for candidate answers corpus ca2
    score_ti_ca2i = calculating_bm25_scores(bug_example['Title'])
    score_di_ca2i = calculating_bm25_scores(bug_example['Description'])
    score_bi_ca2i = calculating_bm25_scores(bug_example['Title+Description'])
    score_qi_ca2i = calculating_bm25_scores(bug_example['Question'])
    score_tdqi_ca2i = calculating_bm25_scores(bug_example['Title+Description+Question'])

    # index ca3 here
    columnname = 'Candidate_Answer_3'
    index_the_corpus(corpus,columnname)
    # scores for candidate answers corpus ca3
    score_ti_ca3i = calculating_bm25_scores(bug_example['Title'])
    score_di_ca3i = calculating_bm25_scores(bug_example['Description'])
    score_bi_ca3i = calculating_bm25_scores(bug_example['Title+Description'])
    score_qi_ca3i = calculating_bm25_scores(bug_example['Question'])
    score_tdqi_ca3i = calculating_bm25_scores(bug_example['Title+Description+Question'])

    for index, row in relevance_score_df.iterrows():

        n = len(corpus.IssueID)
        all_issue_ids = corpus.IssueID.astype('int')

        for i in range(n):
            cas = []
            id = all_issue_ids[i]
            common_sum = row.iloc[i]

            ca1_sum = sum([common_sum, score_bi_ca1i[i], score_qi_ca1i[i], score_di_ca1i[i], score_ti_ca1i[i],
                           score_tdqi_ca1i[i]])
            ca2_sum = sum([common_sum, score_bi_ca2i[i], score_qi_ca2i[i], score_di_ca2i[i], score_ti_ca2i[i],
                           score_tdqi_ca2i[i]])
            ca3_sum = sum([common_sum, score_bi_ca3i[i], score_qi_ca3i[i], score_di_ca3i[i], score_ti_ca3i[i],
                           score_tdqi_ca3i[i]])

            cas = [ca1_sum, ca2_sum, ca3_sum]
            relevance_scores_ca_wise[str(id)] = cas
    return relevance_scores_ca_wise


def all_bm25_scores(bug_example):
    print("calculating all scores...")
    # first is the query, the second is the corpus

    # index title here
    column_name ='Title'
    index_the_corpus(corpus,column_name)
    # calculating scores for title corpus
    score_ti_ti = calculating_bm25_scores(bug_example['Title'])
    score_di_ti = calculating_bm25_scores(bug_example['Description'])
    score_bi_ti = calculating_bm25_scores(bug_example['Title+Description'])
    score_qi_ti = calculating_bm25_scores(bug_example['Question'])
    score_tdqi_ti = calculating_bm25_scores(bug_example['Title+Description+Question'])

    #index the Description here
    column_name = 'Description'
    index_the_corpus(corpus,column_name)
    # calculating scores for description corpus
    score_ti_di = calculating_bm25_scores(bug_example['Title'])
    score_di_di = calculating_bm25_scores(bug_example['Description'])
    score_bi_di = calculating_bm25_scores(bug_example['Title+Description'])
    score_qi_di = calculating_bm25_scores(bug_example['Question'])
    score_tdqi_di = calculating_bm25_scores(bug_example['Title+Description+Question'])

    # index the Question here
    column_name = 'Question'
    index_the_corpus(corpus,column_name)
    # calculating scores for Question corpus
    score_ti_qi = calculating_bm25_scores(bug_example['Title'])
    score_di_qi = calculating_bm25_scores(bug_example['Description'])
    score_bi_qi = calculating_bm25_scores(bug_example['Title+Description'])
    score_qi_qi = calculating_bm25_scores(bug_example['Question'])
    score_tdqi_qi = calculating_bm25_scores(bug_example['Title+Description+Question'])

    # index the Title+Description here
    column_name = 'Title+Description'
    index_the_corpus(corpus,column_name)
    # calculating scores for Title+Description corpus
    score_ti_bi = calculating_bm25_scores(bug_example['Title'])
    score_di_bi = calculating_bm25_scores(bug_example['Description'])
    score_bi_bi = calculating_bm25_scores(bug_example['Title+Description'])
    score_qi_bi = calculating_bm25_scores(bug_example['Question'])
    score_tdqi_bi = calculating_bm25_scores(bug_example['Title+Description+Question'])

    # index the Title+Description+Question here
    column_name = 'Title+Description+Question'
    index_the_corpus(corpus, column_name)
    # The title combinations
    score_ti_tdqi = calculating_bm25_scores(bug_example['Title'])
    score_di_tdqi = calculating_bm25_scores(bug_example['Description'])
    score_bi_tdqi = calculating_bm25_scores(bug_example['Title+Description'])
    score_qi_tdqi = calculating_bm25_scores(bug_example['Question'])
    score_tdqi_tdqi = calculating_bm25_scores(bug_example['Title+Description+Question'])

    relevance_score_matrix_common_sum = sum(
        [score_ti_ti, score_ti_qi, score_ti_di, score_ti_bi, score_bi_bi, score_bi_ti, score_bi_qi, score_bi_di,
         score_qi_qi, score_qi_ti, score_qi_bi, score_qi_di, score_di_di, score_di_bi, score_di_ti, score_di_qi,
         score_tdqi_tdqi, score_tdqi_ti, score_tdqi_di, score_tdqi_qi, score_tdqi_bi, score_ti_tdqi, score_di_tdqi,
         score_bi_tdqi, score_qi_tdqi])

    print("all scores calculated")

    relevance_score_dataframe = pd.DataFrame(relevance_score_matrix_common_sum)
    relevance_score_dataframe = relevance_score_dataframe.T

    return relevance_score_dataframe


def applying_on_dataset(filename):
    # global variables
    global gold
    global corpus
    global issueidslist
    # global filename

    # getting the goldset
    gold = pd.read_csv("../Data/Gold/"+filename+"_goldset.csv")
    gold['Title+Description'] = gold['Title'] + gold['Description']
    gold['Title+Description+Question'] = gold['Title'] + gold['Description'] + gold['Question']
    print(len(gold))

    # getting the corpus
    corpus = pd.read_csv("../Data/Corpus/"+filename+"_corpus.csv")
    corpus = corpus.rename(columns={'Body': 'Description'})
    # print(corpus)
    corpus['Title+Description'] = corpus['Title'] + corpus['Description']
    corpus['Title+Description+Question'] = corpus['Title'] + corpus['Description'] + corpus['Question']
    corpus.IssueID = corpus.IssueID.astype('int')


    # defining the issue list order
    issueidslist = list(corpus.IssueID)
    print(len(issueidslist))

    gold = gold[gold.IssueID.isin(issueidslist)]


    # empty rel dataframe
    relevance_score_dataframe = pd.DataFrame()

    # calling the bm25 score calculator
    for i, row in gold.iterrows():
        # getting bm25 scores
        score = all_bm25_scores(row)
        relevance_score_dataframe = pd.concat([relevance_score_dataframe, score], axis=0, ignore_index=True)


    relevance_score_dataframe.columns = issueidslist

    relevance_score_dataframe.insert(loc=0, column='IssueID', value=gold.IssueID)

    relevance_score_dataframe = relevance_score_dataframe[relevance_score_dataframe.IssueID.isin(issueidslist)]

    print(tabulate(relevance_score_dataframe, headers='keys', tablefmt='psql'))
    print(relevance_score_dataframe)

    for i, row in relevance_score_dataframe.iterrows():
        issue_id = row.IssueID
        relevance_score_dataframe[int(issue_id)][i] = -1000
    intersect = set(relevance_score_dataframe.IssueID).intersection(relevance_score_dataframe.columns)
    first_5 = list(intersect)[:5]

    print(relevance_score_dataframe[relevance_score_dataframe.IssueID.isin(first_5)][['IssueID'] + first_5])


    relevance_score_dataframe.to_csv("../Relevance Scores/"+filename+"_relevance_common_with_issueid.csv",
            encoding='utf-8', index=False)
    relevance_score_dataframe.drop('IssueID', axis=1, inplace=True)

    # common relevance score - without IssueID
    relevance_score_dataframe.to_csv(
        "../Common Relevance Scores/"+filename+"_relevance_common_without_issueid.csv",
        encoding='utf-8',
        index=False)

    relevance_score_dataframe.insert(loc=0, column='IssueID', value=gold.IssueID)
    global relevance_scores_ca_wise_for_every_issue
    relevance_scores_ca_wise_for_every_issue = {}

    for i, row in gold.iterrows():
        print(i)
        issue = row.IssueID
        print(issue)
        df = relevance_score_dataframe.loc[relevance_score_dataframe['IssueID'] == issue]
        df.drop('IssueID', axis=1, inplace=True)
        # calling ca wise rel score
        rel = ca_wise_relevance_scores(row, df, issue)
        file_name = "../Common Relevance Scores Subject System Wise/"+filename+"_relevance_score_subject_systemwise.txt"
        with open(file_name, 'w') as fp:
            json.dump(rel, fp)


def setup_elasticsearch():
    ca_cert = "C:\\Users\\username\\http_ca.crt"
    # setting up elasticsearch
    global es

    es = Elasticsearch(
        hosts='https://localhost:9200',
        ca_certs=ca_cert,
        basic_auth=("elastic", "IxP3lKIBbQEGNv7wrDW-"),
    )
    es.options(ignore_status=[400, 404]).indices.delete(index='bug_report')
    # creating a general mapping for bug reports
    mappings = {
        "properties": {
            "IssueID": {"type": "integer"},
            "Title": {"type": "text", "analyzer": "english"},
            "Description": {"type": "text", "analyzer": "english"},
            "Question": {"type": "text", "analyzer": "english"},
            "Candidate_Answer_1": {"type": "text", "analyzer": "english"},
            "Candidate_Answer_2": {"type": "text", "analyzer": "english"},
            "Candidate_Answer_3": {"type": "text", "analyzer": "english"},
        }
    }
    es.indices.create(index="bug_report", mappings=mappings)


if __name__ == '__main__':


    global filelist
    global name

    filelist = ['filename']

    # running the elasticsearch setup
    setup_elasticsearch()

    for name in filelist:
        print("starting with --------------" + name + "----------------------")
        applying_on_dataset(name)
        print("done with --------------" + name + "----------------------")
        # break