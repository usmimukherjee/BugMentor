import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os

global bertse_embeddings
global tokenizer
from tqdm import tqdm

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def encode_text(text, model, tokenizer):
    """Tokenize text, get embeddings, and return averaged embedding."""
    if not isinstance(text, str) or text.strip() == "":
        return np.zeros((768,))  # Ensure 768-dimensional output

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def calculating_embedding_similarity(name, model, tokenizer):
    # Load goldset data
    goldset_data = pd.read_csv(f"./data/bugmentor_gold/{name}.csv")
    goldset_data["IssueID"] = goldset_data["IssueID"].astype(str)
    
    # Load corpus data
    corpus_data = pd.read_csv(f"./data/bugmentor_corpus_ablation/{name}.csv")

    for _, gold_row in tqdm(goldset_data.iterrows(), total=len(goldset_data), desc=f"Processing {name}"):
        issue_id = gold_row["IssueID"]
        gold_bug_report = gold_row["Title"] + " " + gold_row["Description"]
        gold_question = gold_row["Question"]
        gold_accepted_answer = gold_row["RelevantAnswerDetail"]  # The gold set's accepted answer

        gold_bug_embedding = encode_text(gold_bug_report, model, tokenizer)
        gold_question_embedding = encode_text(gold_question, model, tokenizer)
        
        similarity_scores = []
        
        for _, corpus_row in corpus_data.iterrows():
            corpus_bug_report = corpus_row["Title"] + " " + corpus_row["Description"]
            corpus_question = corpus_row["Question"]
            corpus_answer = corpus_row["Answer"]

            corpus_bug_embedding = encode_text(corpus_bug_report, model, tokenizer)
            corpus_question_embedding = encode_text(corpus_question, model, tokenizer)
            corpus_answer_embedding = encode_text(corpus_answer, model, tokenizer)

            bug_similarity = cosine_similarity([gold_bug_embedding], [corpus_bug_embedding])[0][0]
            question_similarity = cosine_similarity([gold_question_embedding], [corpus_question_embedding])[0][0]
            answer_similarity = cosine_similarity([gold_question_embedding], [corpus_answer_embedding])[0][0]
            
            total_similarity = bug_similarity + question_similarity + answer_similarity  # Combined similarity
            
            similarity_scores.append((corpus_row["IssueID"], corpus_bug_report, corpus_question, corpus_answer, total_similarity))
        
        # Sort by similarity score and get top 5
        similarity_scores.sort(key=lambda x: x[4], reverse=True)
        top_5 = similarity_scores[:5]

        # Prepare results
        results = []
        for result in top_5:
            results.append({
                "IssueID": result[0],
                "GoldBugReport": gold_bug_report,
                "GoldQuestion": gold_question,
                "AcceptedAnswer": gold_accepted_answer,
                "BugReport": result[1],
                "Question": result[2],
                "RelevantAnswerDetail": result[3],
                "similarity_score": result[4]
            })

        # Convert to DataFrame and save for this specific IssueID
        result_df = pd.DataFrame(results)
        result_df.to_csv(f"./results/ablation_embedding/{name}/{issue_id}.csv", index=False)

if __name__ == "__main__":
    # filelist = os.listdir("./data/bugmentor_gold/")
    filelist = ['angular.csv']
    for name in filelist:
        if name in ["readme.md", "motivating.csv"]:
            continue
        
        base_filename, _ = os.path.splitext(name)
        directory_path = "./results/ablation_embedding/" + base_filename + "/"

        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)

        print(f"\nStarting with {base_filename} -------------------")
        calculating_embedding_similarity(base_filename, model, tokenizer)
        print(f"\nFinished with {base_filename} -------------------")