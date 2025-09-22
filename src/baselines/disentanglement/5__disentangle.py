import pandas as pd
import numpy as np
import nltk
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import re
import string
import os
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning) 
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
global tokenizer
global model
global model_dir
model_dir = './BugMentor/models/torch-bert-uncased-checkpoint/'

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

class DialBERT(nn.Module):
    def __init__(self, model):
        super(DialBERT, self).__init__()
        # Initialize the BertModel from a TensorFlow checkpoint
        self.bert = model
        self.bi_lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.bi_lstm(sequence_output)
        return lstm_output

def tokenize_input(text, tokenizer):
    return tokenizer(str(text), padding='max_length', truncation=True, max_length=512, return_tensors='pt')

def compute_similarities(question, comments, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DialBERT(model).to(device)

    question_tokens = tokenize_input(question, tokenizer)
    question_ids = question_tokens['input_ids'].to(device)
    question_mask = question_tokens['attention_mask'].to(device)

    model.eval()  # Put the model in evaluation mode
    with torch.no_grad():  # Turn off gradients to save memory and computations
        question_features = model(question_ids, question_mask).mean(dim=1)

    similarities = []
    cos = nn.CosineSimilarity(dim=1)
    for comment in comments:
        comment_tokens = tokenize_input(comment, tokenizer)
        comment_ids = comment_tokens['input_ids'].to(device)
        comment_mask = comment_tokens['attention_mask'].to(device)

        with torch.no_grad():
            comment_features = model(comment_ids, comment_mask).mean(dim=1)
        similarity = cos(question_features, comment_features)
        similarities.append(similarity.item())

    return similarities

def disentanglement(question, comment_df):
    print("disentangling comments")
    comments = comment_df.Comment.tolist()
    similarities = compute_similarities(question, comments, tokenizer, model)
    sorted_comments = [x for _, x in sorted(zip(similarities, comments), reverse=True)]
    print("disentangle done")
    return sorted_comments
        
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
       
def applying_on_dataset(filename):
    global gold
    global corpus
    gold = pd.read_csv("./BugMentor/data/bugmentor_gold/"+filename)
    gold.IssueID = gold.IssueID.astype('str')

    corpus = pd.read_csv("./BugMentor/data/2__bug_reports_comments/"+filename,encoding='iso-8859-1',dtype=str, low_memory=False)
    corpus = corpus.rename(columns={'Body': 'Description'})
    corpus['Comment'] = corpus['Comment'].fillna('')
    corpus['Comment'] = corpus['Comment'].apply(str)
    
    print(corpus.columns)
    corpus.IssueID = corpus.IssueID.astype('str')
    issueidslist = list(corpus.IssueID)
    
    base_filename = ''
    base_filename, extension = os.path.splitext(filename)
    issues_completed_files =  os.listdir("./BugMentor/results/baselines/disentangle/"+base_filename+'/')
    issues_completed = [str(os.path.splitext(file)[0]) for file in issues_completed_files]
    base_filename = ''
    
    # iterating over each row of gold dataframe
    for index, row in gold.iterrows():
        if str(row.IssueID) in issues_completed:
            continue
        query = row['Question']
        results = disentanglement(str(query),corpus)
        ground_truth = row['RelevantAnswerDetail']
        # save the results and ground truth in a csv file as a dataframe
        results_df = pd.DataFrame(results, columns=['Result'])
        results_df['IssueID'] = row.IssueID
        results_df['Title'] = row.Title
        results_df['Description'] = row.Description
        results_df['Question'] = row.Question
        results_df['GroundTruth'] = ground_truth
        
        base_filename, extension = os.path.splitext(filename)
        name_path = "./BugMentor/results/baselines/disentangle/"+base_filename+'/'+str(row.IssueID)+'.csv'
        results_df.to_csv(name_path, index=False)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('./BugMentor/models/tokenizer/')
    print("tokenizer loaded")
    model = BertModel.from_pretrained(model_dir)
    print("model loaded")
    filelist = ['axios.csv']
    for name in filelist:
        if name == 'readme.md':
            continue
        base_filename, extension = os.path.splitext(name)
        directory_path = "./BugMentor/results/baselines/disentangle/"+base_filename
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
        print('doing '+ str(name))
        applying_on_dataset(name)
        print('done '+ str(name))