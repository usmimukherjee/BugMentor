
import numpy as np
import pandas as pd
import re
import string
import os
import nltk
nltk.download('wordnet')
# wmd
from word_mover_distance import model
# bleu
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
# meteor
from nltk.translate import meteor
from nltk import word_tokenize
import nltk
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def calculate_meteor(candidate, reference):
    reference = word_tokenize(reference)
    candidate = word_tokenize(candidate)
    meteor_score = round(meteor([candidate], reference), 4)
    return meteor_score

""" Normalized Smooth BLEU """

def normalized_smooth_bleu_score(text1, text2):
    bleu_score = sentence_bleu([word_tokenize(text1)], word_tokenize(text2))
    max_bleu = 1
    return bleu_score / max_bleu

""" Semantic Similarity """

def calculate_semantic_similarity(s1, s2):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings1 = embedding_model.encode(s1, convert_to_tensor=True)
    embeddings2 = embedding_model.encode(s2, convert_to_tensor=True)

    cosine_scores2 = util.cos_sim(embeddings1, embeddings2)

    return cosine_scores2[0][0].item()

def evaluation_data(my_model,name):
    path = "./BugMentor/results/baselines/disentangle/"+name+"/"
    files = os.listdir(path)
    for file in files:
        if file == 'readme.md':
            continue
        print("starting with " + file + " -------------------")
        eval_data = pd.read_csv(path + file)
        # get subset of 10 rows
        eval_data = eval_data.head(10)
        # eval_data = pd.read_csv(path + name + ".csv")
        eval_data['BLEU_SCORE'] = ''
        eval_data['SEMANTIC_SIMILARITY'] = ''
        eval_data['METEOR_SCORE'] = ''
        eval_data['WMD_SCORE'] = ''

        for index, row in tqdm(eval_data.iterrows(), total=eval_data.shape[0], desc="Evaluating"):
            ground_truth = row['GroundTruth']
            generated_answer = row['Result']
            # Calculate scores
            bleu_score = normalized_smooth_bleu_score(str(ground_truth), str(generated_answer))
            semantic_similarity = calculate_semantic_similarity(str(ground_truth), str(generated_answer))
            meteor_score = calculate_meteor(str(ground_truth), str(generated_answer))
            wmd_score = my_model.wmdistance(str(ground_truth), str(generated_answer))
            
            eval_data.at[index, 'BLEU_SCORE'] = bleu_score*100
            eval_data.at[index, 'SEMANTIC_SIMILARITY'] = semantic_similarity*100
            eval_data.at[index, 'METEOR_SCORE'] = meteor_score
            eval_data.at[index, 'WMD_SCORE'] = wmd_score    
        
            final_path = "./BugMentor/evaluated/baselines/disentangle/"+name+"/"+file
            eval_data.to_csv(final_path, index=False)

if __name__ == '__main__':
    
    filelist = ['angular.csv']
    my_model = model.WordEmbedding(model_fn="./BugMentor/models/glove/glove.6B.300d.txt")
    for name in filelist:
        if name == 'readme.md':
            continue
        base_filename, extension = os.path.splitext(name)
        directory_path = "./BugMentor/evaluated/baselines/disentangle/"+base_filename
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
        evaluation_data(my_model,base_filename)
        
        print('done '+base_filename)