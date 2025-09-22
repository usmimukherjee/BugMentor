import transformers
import pandas as pd
import warnings
import torch
import os
from torch import cuda, bfloat16
import urllib.request
import re
import string
from nltk import word_tokenize
import nltk
from tqdm import tqdm
warnings.filterwarnings('ignore')
import logging
from sentence_transformers import SentenceTransformer, util
import sys
from transformers import pipeline
global device
global model, embeddings, tokenizer
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


def init_model_embeddings_tokenizer():

    # load model
    model_id = 'mistralai/Mistral-7B-v0.1'
    # begin initializing HF items, you need an access token
    hf_auth = ''

    # model config
    model_config = transformers.AutoConfig.from_pretrained(model_id,use_auth_token=hf_auth)

    # mem config
    bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True,
                                                bnb_4bit_quant_type='nf4',
                                                bnb_4bit_use_double_quant=True,
                                                bnb_4bit_compute_dtype=bfloat16)

    # load model
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id,
                                                            trust_remote_code=True,
                                                            quantization_config=bnb_config,
                                                            config=model_config,
                                                            token=hf_auth)

    # inference mode
    model.eval()

    print(f"Model loaded on {device}")

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
    )

    # load embedding model
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cuda"}
    embeddings = SentenceTransformer(embedding_model_name, device=device)
    print("loaded embeddings")
    return model, embeddings, tokenizer

model, embeddings, tokenizer = init_model_embeddings_tokenizer()
global pipe
pipe = pipeline("text-generation", model=model,tokenizer = tokenizer,torch_dtype=torch.bfloat16,device_map="auto")


# For the baseline we use the T5 model using the context as just the provided bug report for the question
def get_answers(name):
    gold = pd.read_csv("./data/bugmentor_gold/" + name + ".csv")
    gold.IssueID = gold.IssueID.astype('str')
    goldsetids = list(gold.IssueID)
    
    base_filename, extension = os.path.splitext(name)

    for iterator, ids in enumerate(tqdm(goldsetids, desc="Processing Issues", unit="file")):
        filename = f"./results/ablation_embedding/{base_filename}/{ids}.csv"
        
        if not os.path.exists(filename):
            print(f"File {filename} not found, skipping...")
            continue  
        
        file = pd.read_csv(filename)
        dataframe = pd.DataFrame(file)
        dataframe['MistralAnswer'] = " "

        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Processing {ids}", unit="row", leave=False):
            question = row.Question
            bug_report = row['Title'] + row['Description']
            relevant_candidate_answer = row['Candidate_Answer']
            prompt = f"""Answer Question on the bug report based on the relevant information.
            Here is a bug report which has incomplete information.
            ## Bug Report - \n{bug_report}
            There is a follow-up question asking for missing information.
            \nHere is some relevant information from previous bug report.\n{relevant_candidate_answer}\n
            Can you answer the question below based on the bug report?
            Here is the question.\n 
            ## Question- \n{question}\n ## Answer : \n """
            sequences = pipe( prompt, do_sample=True, max_new_tokens=100, 
                temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=1,)
            dataframe.at[index, 'MistralAnswer'] = sequences[0]['generated_text']
        
        base_filename, extension = os.path.splitext(name)
        name_path = "../results/ablation_embedding_mistral/"+base_filename+'.csv'
        dataframe.to_csv(name_path, index=False)


if __name__ == '__main__':
    filelist = os.listdir("../../results/ablation_embedding/")
    for name in filelist:
        base_filename, extension = os.path.splitext(name)
        print('doing '+name)
        get_answers(name)
        print('done '+name)
