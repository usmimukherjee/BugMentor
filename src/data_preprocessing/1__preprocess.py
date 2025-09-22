import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def pre_process(issuefilename,commentfilename,name):
    # method to merge bug reprts and comments
    
    # bug reports --------------
    issue_reports = pd.read_csv(issuefilename,encoding='iso-8859-1',dtype=str, low_memory=False)
    issue_reports.rename(columns={'id': 'IssueID', 'Body': 'Description'}, inplace=True)
    issue_reports.IssueID = issue_reports.IssueID.astype(np.int64)
    # print(issue_reports)
    
    # issue comments -------------
    comments = pd.read_csv(commentfilename,encoding='iso-8859-1',dtype=str, low_memory=False)
    comments = comments[(comments['IssueNumber'] != 'IssueNumber')]
    comments.rename(columns={'IssueNumber': 'IssueID'}, inplace=True)
    comments['IssueID'] = pd.to_numeric(comments['IssueID'], errors='coerce')
    # print(comments.info())
    # print(issue_reports.info())
    
    # merge bug reports and comments ------------
    merged_issues = pd.merge(issue_reports, comments,how='inner',on=["IssueID"])
    # print(merged_issues.columns)

    # save --------------
    merged_issues.to_csv("../../data/corpus/"+name,encoding='utf-8', index=False)
    
def main():
    bug_reports_dir = "../../data/complete_bug_reports/"
    issue_comments_dir = "../../data/complete_issue_comments/"
    files = os.listdir(bug_reports_dir)
    # run with tqdm to see progress
    for file in files:
        print("Preprocessing for ", file)
        bug_reports = bug_reports_dir + file
        issuecomments = issue_comments_dir + "comments_"+ file
        pre_process(bug_reports, issuecomments,file)
        print("Preprocessing done for ", file)


if __name__ == '__main__':
    main()