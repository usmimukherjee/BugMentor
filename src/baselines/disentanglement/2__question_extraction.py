import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
import re
import string
import warnings
warnings.filterwarnings("ignore")

global question_df
global dictionary_of_question_answers
global all_questions
global all_issue_ids


# code for detecting questions from issue comments using nltk

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

def is_ques_using_nltk(ques):
    question_type = classifier.classify(dialogue_act_features(ques))
    return question_type in question_types

# check with custom pipeline if still this is a question mark it as a question
def is_question(question):
    question = question.lower().strip()
    if not is_ques_using_nltk(question):
        is_ques = False
        # check if any of pattern exist in sentence
        for pattern in question_pattern:
            is_ques  = pattern in question
            if is_ques:
                break

        # there could be multiple sentences so divide the sentence
        sentence_arr = question.split(".")
        for sentence in sentence_arr:
            if len(sentence.strip()):
                # if question ends with ? or start with any helping verb
                # word_tokenize will strip by default
                first_word = nltk.word_tokenize(sentence)[0]
                if sentence.endswith("?") or first_word in helping_verbs:
                    is_ques = True
                    break
        return is_ques
    else:
        return True

def filter_qa(filename):

    global question_df
    question_df = pd.DataFrame()
    global dictionary_of_question_answers
    dictionary_of_question_answers = {}
    corpus =  pd.read_csv("../../data/fetched_bugreports/"+filename,encoding='utf-8',dtype=str, low_memory=False)
    corpus.IssueID = corpus.IssueID.astype(str)
    unique_issue_id = list(np.unique(corpus.IssueID))
    
    # grouping the corpus based on the issue id
    grouped = corpus.groupby(corpus.IssueID)

    # list of all the questions and candidate answers
    all_questions = []
    all_issue_ids = []

    for i, v in enumerate(unique_issue_id): 
        # i is index, v is group-key (issue-id)

        # for every issue it creates a dataframe sorted based on the time
        each_issue_df = (grouped.get_group(unique_issue_id[i]).sort_values(by='CommentTime')).reset_index(drop=True)
        each_issue_df.dropna(inplace=True)
        # print(f'For issue: {v}, ({len(each_issue_df)})')

        # if the issue only contains one comment
        if(len(each_issue_df) == 1 or (len(each_issue_df)==0)):
            continue
        # iterating through every comment of every issue dataframe
        for index, row in each_issue_df.iterrows():

            # checking if the comment is a question or not and also checking if the commenter is not the author of the bug report
            if is_question(row.Comment) == True and row.CommentUSER != row.Author:
                # print(v, index, row.Comment, 'is question')
                question_df = pd.concat([question_df, pd.DataFrame([row])], ignore_index=True)
                dictionary_of_question_answers[row.IssueID] = question_df.Comment
                question = row.Comment
                issue_id = row.IssueID
                all_issue_ids.append(row.IssueID)
                all_questions.append(question)
    questions_and_candidate_answers_dataframe = pd.DataFrame({
        'IssueID': all_issue_ids,
        'Question': all_questions,
    })

    questions_and_candidate_answers_dataframe.to_csv("../../data/questions/"+filename, encoding='utf-8', index=False)
 
                
if __name__ == '__main__':
    
    # question types and helping verbs declaration
    question_types = ["whQuestion", "ynQuestion"]

    question_pattern = [ "do you", "what", "is it", "why", "would you", "how", "is there",
                        "are there", "is it so", "is this true", "to know", "is that true", "are you",
                        "question is", "tell me more", "can you", "tell me", "can you explain",
                        "question", "answer", "questions", "answers", "ask"]
    helping_verbs = ["is", "am", "can", "are", "do", "does"]

    posts = nltk.corpus.nps_chat.xml_posts()[:10000]
    featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
    # 10% of the total data
    size = int(len(featuresets) * 0.1)
    # first 10% for test_set to check the accuracy, and rest 90% after the first 10% for training
    train_set, test_set = featuresets[size:], featuresets[:size]
    # get the classifer from the training set
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    # filelist = os.listdir('../../data/fetched_bugreports/')
    filelist = ['ansible.csv']
    for name in filelist:
        if name == 'readme.md':
            continue
        print('doing '+name)
        filter_qa(name)
        print('done '+name)
