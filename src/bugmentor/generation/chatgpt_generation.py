from openai import OpenAI
import os
from tqdm import tqdm
import pandas as pd

os.environ["OPENAI_API_KEY"] = ""
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_answers(name):
    gold = pd.read_csv("./data/bugmentor_gold/" + name + ".csv")
    gold.IssueID = gold.IssueID.astype('str')
    goldsetids = list(gold.IssueID)
    
    base_filename, extension = os.path.splitext(name)

    for iterator, ids in enumerate(tqdm(goldsetids, desc="Processing Issues", unit="file")):
        filename = f"./results/bm25_embedding/{base_filename}/{ids}.csv"
        
        if not os.path.exists(filename):
            print(f"File {filename} not found, skipping...")
            continue  
        
        file = pd.read_csv(filename)
        dataframe = pd.DataFrame(file)
        dataframe['ChatGPTAnswer'] = " "

        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Processing {ids}", unit="row", leave=False):
            question = row.Question
            bug_report = row['GoldBugReport']
            relevant_candidate_answer = row['RelevantAnswerDetail']
            prompt = f"""Answer Question on the bug report based on the relevant information.
            Here is a bug report which has incomplete information.
            ## Bug Report - \n{bug_report}
            There is a follow-up question asking for missing information.
            \nHere is some relevant information from previous bug report.\n{relevant_candidate_answer}\n
            Can you answer the question below based on the bug report?
            Here is the question.\n 
            ## Question- \n{question}\n ## Answer : \n """

            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            dataframe.at[index, 'ChatGPTAnswer'] = completion.choices[0].message.content        
        
        output_dir = f"./results/bm25_embedding_chatgpt/{base_filename}"
        os.makedirs(output_dir, exist_ok=True)

        name_path = f"{output_dir}/{ids}.csv"
        dataframe.to_csv(name_path, index=False)


if __name__ == '__main__':
    filelist = os.listdir('./results/bm25_embedding/')
    for name in filelist:
        if name == 'readme.md' or name == 'motivating' or name == 'angular' or name == 'sympy':
            continue
        base_filename, extension = os.path.splitext(name)
        print('starting with file-------'+name)
        get_answers(name)
        print('completed with file-------'+name)
