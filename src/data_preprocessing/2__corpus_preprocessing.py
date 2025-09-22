import pandas as pd
import numpy as np
import nltk
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import re
import os
import string
import warnings
warnings.filterwarnings("ignore")

# some variable declaration
global question_tokens
global answers_tokens
global question_df
global comment_reply_by_author
global dictionary_of_question_answers
# list of all the questions and candidate answers
global all_questions
global all_candidate_answers1
global all_candidate_answers2
global all_candidate_answers3
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

def tokenise_this(sentences):
    tokens = []
    words = word_tokenize(sentences)
    tokens.append(words)
    return tokens

def tokenise_this2(sentences):
    tokens = []
    for s in sentences:
        words = word_tokenize(s)
        tokens.append(words)
    return tokens

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

def find_candidate_answer3(question, answers_df):
    # print("question length : ",len(question))
    # print("answers df length: ",len(answers_df))

    # if length of the possible answers dataframe is zero it will return null
    if len(answers_df) == 0:
        return "N/A"
    elif len(answers_df) == 1:
        return str(answers_df.Comment)

    # cleaning the comment
    # answers_df['Comment'] = answers_df['Comment']
    #
    # # cleaning the question
    # question = clean_text_round1(question)

    # tokenizing the question
    question_tokens = tokenise_this(question)

    # making a list of all the answer comments
    answers_comments = answers_df.Comment

    # tokenizing the answer comments
    answers_tokens = tokenise_this2(answers_comments)

    # running the BM25 on the answer comments
    bm25 = BM25Okapi(answers_tokens)

    # getting the top answer similar to the question from the answer comments
    candidate_answer3 = bm25.get_top_n(question_tokens[0], answers_comments.tolist(), n=1)

    # returning the top answer as the candidate answer
    return candidate_answer3

def filter_qa(filename):

    global question_df
    question_df = pd.DataFrame()
    global dictionary_of_question_answers
    dictionary_of_question_answers = {}
    corpus =  pd.read_csv("../../data/2__bug_report_comments/"+filename)

    unique_issue_id = list(np.unique(corpus.IssueID))
    # print(unique_issue_id[60:])

    grouped = corpus.groupby(corpus.IssueID)
    # print(grouped.get_group(unique_issue_id[10]))

    # list of all the questions and candidate answers
    all_questions = []
    all_candidate_answers1 = []
    all_candidate_answers2 = []
    all_candidate_answers3 = []
    all_issue_ids = []

    for i, v in enumerate(unique_issue_id):  # iterating through each issue
        # i is index, v is group-key (issue-id)

        # for every issue it creates a dataframe sorted based on the time
        each_issue_df = (grouped.get_group(unique_issue_id[i]).sort_values(by='CommentTime')).reset_index(drop=True)
        # print(each_issue_df)
        each_issue_df.dropna(inplace=True)
        # each_issue_df = each_issue_df[
        #     each_issue_df.apply(lambda row: is_question(row.Comment) == True and row.CommentUSER != row.Author, axis=1)
        # ]
        print(f'For issue: {v}, ({len(each_issue_df)})')

        # if the issue only contains one comment
        if(len(each_issue_df) == 1 or (len(each_issue_df)==0)):
            continue
        # iterating through every comment of every issue dataframe
        for index, row in each_issue_df.iterrows():

            # checking if the comment is a question or not and also checking if the commenter is not the author of the bug report
            if is_question(row.Comment) == True and row.CommentUSER != row.Author:
                # print(v, index, row.Comment, 'is question')

                question_df = question_df.append(row, ignore_index=True)
                dictionary_of_question_answers[row.IssueID] = question_df.Comment
                question = row.Comment
                issue_id = row.IssueID

                # the first answer after the question
                subsequent_comments_df = each_issue_df.iloc[index + 1:]

                if not subsequent_comments_df.empty:
                    # print(v, index, ' has as subsequent_comments_df')
                    # first candidate answer
                    candidate_answer1 = subsequent_comments_df.iloc[0]
                    subsequent_comments_df = subsequent_comments_df[
                        subsequent_comments_df.CommentID != candidate_answer1.CommentID]

                    # first answer by the author after the question
                    # comment_reply_by_author_moby = comment_reply_by_author_moby.append(subsequent_comments_df[(subsequent_comments_df['CommentUSER'] == subsequent_comments_df['Author'] )],ignore_index=True)

                    # second candidate answer
                    # if not comment_reply_by_author_moby.empty:
                    replies_by_author = subsequent_comments_df[
                        (subsequent_comments_df['CommentUSER'] == subsequent_comments_df['Author'])
                    ]
                    if not replies_by_author.empty:
                        candidate_answer2 = replies_by_author.iloc[0]
                        subsequent_comments_df = subsequent_comments_df[
                            subsequent_comments_df.CommentID != candidate_answer2.CommentID]
                    else:
                        candidate_answer2 = None

                    # third candidate answer
                    candidate_answer3 = find_candidate_answer3(question, subsequent_comments_df)
                    # print(candidate_answer3,candidate_answer1.empty,candidate_answer2.empty)

                    if (candidate_answer3 or candidate_answer3!='N' or not candidate_answer3.isnumeric())and not candidate_answer1.empty:

                        # print(row.IssueID)
                        all_issue_ids.append(row.IssueID)

                        # print("\n question: \n",question)
                        all_questions.append(question)

                        # print("\n candidate_answer 1: \n",candidate_answer1.Comment)
                        all_candidate_answers1.append(candidate_answer1.Comment if candidate_answer1 is not question else None)

                        # print("\n candidate_answer 2: \n",candidate_answer2.Comment)
                        all_candidate_answers2.append(
                            candidate_answer2.Comment if candidate_answer2 is not None else 'N/A')

                        # print("\n candidate_answer 3: \n",candidate_answer3[0])
                        all_candidate_answers3.append(candidate_answer3[0])
                    else:
                        print({'candidate_answer3': bool(candidate_answer3),
                               'candidate_answer1': not candidate_answer1.empty,
                               'candidate_answer2': candidate_answer2 is not None})
                else:
                    # print('subsequent_comments_df empty:', v)
                    pass

    print(len(all_issue_ids), len(all_questions), len(all_candidate_answers1), len(all_candidate_answers3))
    print(np.unique(all_issue_ids))

    questions_and_candidate_answers_dataframe = pd.DataFrame({
        'IssueID': all_issue_ids,
        'Question': all_questions,
        'Candidate_Answer_1': all_candidate_answers1,
        'Candidate_Answer_2': all_candidate_answers2,
        'Candidate_Answer_3': all_candidate_answers3,
    })

    print(questions_and_candidate_answers_dataframe)
    questions_and_candidate_answers_dataframe.to_csv("../../data/bugmentor_corpus/"+filename, encoding='utf-8', index=False)


if __name__ == '__main__':



    # question types and helping verbs declaration
    question_types = ["whQuestion", "ynQuestion"]

    question_pattern = [ "do you", "what", "is it", "why", "would you", "how", "is there",
                        "are there", "is it so", "is this true", "to know", "is that true", "are you",
                        "question is", "tell me more", "can you", "tell me", "can you explain",
                        "question", "answer", "questions", "answers", "ask", "please"]
    helping_verbs = ["is", "am", "can", "are", "do", "does"]

    posts = nltk.corpus.nps_chat.xml_posts()[:10000]
    featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
    # 10% of the total data
    size = int(len(featuresets) * 0.1)
    # first 10% for test_set to check the accuracy, and rest 90% after the first 10% for training
    train_set, test_set = featuresets[size:], featuresets[:size]
    # get the classifer from the training set
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    filelist = os.listdir('../../data/2__bug_report_comments/')
    for name in filelist:
        filter_qa(name)
        print('done '+name)
        # break
