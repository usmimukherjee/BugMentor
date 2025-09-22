import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os

global bertse_embeddings
global tokenizer

def calculating_embedding_similarity(name, bertse_embeddings, tokenizer):
    
    goldset_data =  pd.read_csv("../../data/bugmentor_gold/"+name+".csv")
    goldset_data.IssueID = goldset_data.IssueID.astype(str)
    goldsetids = list(goldset_data.IssueID)
    iterator = 0

    for ids in goldsetids:

        filename = ("../../results/bm25/rank_dataframes/"+name+"/"+ids+".csv")
        file = pd.read_csv(filename)

        dataframe = pd.DataFrame(file)
        dataframe['Embedding_Similarity_Score'] = ""
        print(f'\r{iterator} out of {len(goldsetids) - 1}', end='')
        iterator = iterator + 1
        
        for index, row in dataframe.iterrows():
            answer = str(row.RelevantAnswerDetail)
            # Tokenize the input text
            answer_tokens = tokenizer.tokenize(answer)
            answer_ids = tokenizer.convert_tokens_to_ids(answer_tokens)  # Convert tokens to indices
            # Convert indices to a tensor
            answer_tensor = torch.tensor([answer_ids])
            # Get embeddings
            embedded_answer = bertse_embeddings(answer_tensor)
            # Average the embeddings across the token dimension
            embedded_answer_avg = torch.mean(embedded_answer, dim=1)  # Assuming dim=1 is the sequence length dimension

            question = str(row.Question)
            # Tokenize the input text
            question_tokens = tokenizer.tokenize(question)
            question_ids = tokenizer.convert_tokens_to_ids(question_tokens)  # Convert tokens to indices
            # Convert indices to a tensor
            question_tensor = torch.tensor([question_ids])
            # Get embeddings
            embedded_question = bertse_embeddings(question_tensor)
            # Average the embeddings across the token dimension
            embedded_question_avg = torch.mean(embedded_question, dim=1)  # Assuming dim=1 is the sequence length dimension

            # Compute cosine similarity on the averaged embeddings
            sim = cosine_similarity(embedded_answer_avg.detach().numpy(), embedded_question_avg.detach().numpy())
            # print(sim)
            dataframe.at[index, 'Embedding_Similarity_Score'] = sim[0][0]
        
        sorted_df = dataframe.sort_values(by='Embedding_Similarity_Score', ascending=False)
        sorted_df.to_csv("../../results/bm25_embedding/"+name+"/"+ids+".csv", index=False)
        

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    embeddings = model.get_input_embeddings()

    filelist = os.listdir("../../results/bm25/rank_dataframes/")
    # filelist = ["motivating.csv"]
    for name in filelist:
        if name == 'readme.md' :
            continue
        base_filename, extension = os.path.splitext(name)
        directory_path = "../../results/bm25_embedding/"+base_filename
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
        print("\nstarting with " + base_filename + " -------------------")
        calculating_embedding_similarity(base_filename, embeddings, tokenizer)
        print("\nfinished with " + base_filename + " -------------------")