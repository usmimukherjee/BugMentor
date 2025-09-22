# from transformers import T5Tokenizer, AutoModelWithLMHead, pipeline, AutoModelForSeq2SeqLM
# from transformers import BertForQuestionAnswering, AutoModelForTokenClassification
# from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, util

import pandas as pd
import warnings
import os
import urllib.request
import numpy as np
import re
import string
import nltk
# nltk.download('wordnet')
# nltk.download('punkt')

# wmd
from word_mover_distance import model

# bleu
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# meteor
from nltk.translate import meteor

# lemmatize library
from nltk.stem import WordNetLemmatizer

# tokenize library
from nltk import word_tokenize


# ignoring warnings
warnings.filterwarnings('ignore')


# import logging
# import sys


# evaluation metrics------
def calculate_meteor(candidate, reference):
    reference = word_tokenize(str(reference))
    candidate = word_tokenize(str(candidate))
    meteor_score = round(meteor([candidate], reference), 4)
    return meteor_score


def evaluate_wmd(name, my_model, dataframe):
    for index, row in dataframe.iterrows():
        s1 = str(row.AnswerBotSummary)
        s2 = str(row.AcceptedAnswer)
        s1 = s1.lower().split()
        s2 = s2.lower().split()
        wmdistance2 = my_model.wmdistance(s1, s2)
        dataframe.at[index, 'WMD_Score'] = wmdistance2
    return dataframe 


def evaluate_bleu(name, model,dataframe):
    for index, row in dataframe.iterrows():
        s1 = str(row.AnswerBotSummary)
        s2 = str(row.AcceptedAnswer)
        s1 = s1.lower().split()
        s2 = s2.lower().split()
        bleu_score2 = sentence_bleu(s2, s1, smoothing_function=model.method2, weights=(0.25, 0.25, 0.25, 0.25))
        dataframe.at[index, 'BLEU_Score'] = bleu_score2
        # print(bleu_score1,bleu_score2)
    return dataframe


def evaluate_meteor(name,dataframe):
    for index, row in dataframe.iterrows():
        s1 = str(row.AnswerBotSummary)
        s2 = str(row.AcceptedAnswer)
        s1 = s1.lower().split()
        s2 = s2.lower().split()
        score2 = calculate_meteor(s1, s2)
        dataframe.at[index, 'METEOR_Score'] = score2
        # print(score1, score2)
    return dataframe


def evaluate_sbert_score(name,dataframe):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for index, row in dataframe.iterrows():
        s1 = str(row.AnswerBotSummary)
        s2 = str(row.AcceptedAnswer)
        s1 = s1.lower().split()
        s2 = s2.lower().split()

        embeddings1 = model.encode(s1, convert_to_tensor=True)
        embeddings2 = model.encode(s2, convert_to_tensor=True)

        cosine_scores2 = util.cos_sim(embeddings1, embeddings2)

        dataframe.at[index, 'SBERT_Score'] = cosine_scores2[0][0].item()
        # print(cosine_scores1[0][0].item(), cosine_scores2[0][0].item())
    return dataframe


if __name__ == '__main__':
    
    filenames = ["cpp","java","js","python"]

    for name in filenames:
        print("Starting with " + name + " -------------------")
        
        my_model = model.WordEmbedding(model_fn="glove.6B.300d.txt")
        chencherry = SmoothingFunction()
        
        result = pd.read_csv("./AnswerBotResults-Top1/" + name + "_answerbot.csv")
        

        dataframe = pd.DataFrame()
        dataframe = evaluate_wmd(name,my_model,result)
        dataframe = evaluate_bleu(name, SmoothingFunction(), dataframe)
        dataframe = evaluate_meteor(name,dataframe)
        dataframe = evaluate_sbert_score(name,dataframe)

        print("\n finished with " + name + " -------------------")
        
        dataframe.to_csv("./CrossProject/EvaluationMetrics-Top1/" + name + "_answerbot.csv", index=False)

