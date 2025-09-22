import pandas as pd
import numpy as np
import os
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")

def merge_df(filename):
    print(filename+" in progress...")
    issue_reports = pd.read_csv("../../data/fetched_bugreports/"+filename,encoding='utf-8',dtype=str, low_memory=False)
    issue_reports.IssueID = issue_reports.IssueID.astype(str)

    qa = pd.read_csv("../../data/questions/" +filename, encoding='utf-8',dtype=str, low_memory=False)
    qa.IssueID = qa.IssueID.astype(str)

    merged_issues = pd.merge(issue_reports, qa, how='right', on=["IssueID"])
    complete_sorted = merged_issues.sort_values(by='Created At').reset_index(drop=True)
    # drop rows where the comment is same as the question
    complete_sorted = complete_sorted[complete_sorted['Comment'] != complete_sorted['Question']]
    complete_sorted.to_csv("../../data/corpus/" +filename, encoding='utf-8',index=False)

if __name__ == '__main__':
    # filelist =  os.listdir("../../data/questions/")
    filelist = ['ansible.csv']
    for name in filelist:
        print('doing '+name)
        merge_df(name)
        print('done '+name)