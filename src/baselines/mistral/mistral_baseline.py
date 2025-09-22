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
warnings.filterwarnings('ignore')
import logging
from sentence_transformers import SentenceTransformer, util
import sys
global device
global model, embeddings, tokenizer
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


def init_model_embeddings_tokenizer():

    # load model
    model_id = 'mistralai/Mistral-7B-Instruct-v0.2'
    # begin initializing HF items, you need an access token
    hf_auth = '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'

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
# For the baseline we use the T5 model using the context as just the provided bug report for the question
def get_answers(name):
    gold = pd.read_csv("./BugMentor/data/bugmentor_gold/"+name)
    gold.IssueID = gold.IssueID.astype('str')
    goldsetids = list(gold.IssueID)
    iterator = 0

    for ids in goldsetids:
        
        filename = ("./BugMentor/data/bugmentor_gold/" + name)
        file = pd.read_csv(filename)
        
        dataframe = pd.DataFrame(file)
        dataframe['BaselineMistralAnswer'] = " "
        print(f'\r{iterator} out of {len(goldsetids) - 1}', end='')
        iterator += 1

        for index, row in dataframe.iterrows():
            question = row.Question
            context = row['Title'] + row['Description']
            content = context + "  QUESTION:" + question
            messages = [
                {"role": "user", "content": "You are a helpful bot who reads context of software bug reports and answers questions"},
                {"role": "assistant", "content": str(content)},
                {"role":"user", "content": str(question)}
            ]
            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt",return_attention_mask=True, padding = True)
            model_inputs = encodeds.to(device)
            generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
            decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            dataframe.at[index, 'BaselineMistralAnswer'] = decoded
        
        base_filename, extension = os.path.splitext(name)
        name_path = "./BugMentor/results/baselines/mistral/"+base_filename+'/'+str(row.IssueID)+'.csv'
        dataframe.to_csv(name_path, index=False)


if __name__ == '__main__':
    filelist = os.listdir("../../../data/bugmentor_gold/")
    for name in filelist:
        if name == 'readme.md':
            continue
        base_filename, extension = os.path.splitext(name)
        directory_path = "./BugMentor/results/baselines/mistral/"+base_filename
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
        print('doing '+name)
        get_answers(name)
        print('done '+name)
