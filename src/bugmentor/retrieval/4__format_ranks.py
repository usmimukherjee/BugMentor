import pandas as pd
import os

def format_ranks(name):
    # this function will take the ranks of the top 10 candidates and will add the relevant answer and bug report to the dataframe
    complete_data = pd.read_csv("../../data/bugmentor_corpus/"+name+".csv",encoding='iso-8859-1',dtype=str, low_memory=False)
    goldset_data = pd.read_csv("../../data/bugmentor_gold/"+name+".csv")
    goldsetids = set(goldset_data.IssueID)
    goldsetids = list(goldsetids)

    for ids in goldsetids:
        file = pd.read_csv("../../results/bm25/ranks/"+name+"/"+str(ids)+".txt")
        dataframe = pd.DataFrame(file)
        dataframe['RelevantAnswerDetail'] = ""
        dataframe['BugReport'] = ""

        for index, row in dataframe.iterrows():
            df = complete_data.loc[complete_data['IssueID'] == str(file['top-candidate-scorer-id'][index][:-2])]

            if int(file['top-candidate-scorer-id'][index][-1:]) == 1:
                dataframe.at[index, 'RelevantAnswerDetail'] = df.Candidate_Answer_1.values[0]
                dataframe.at[index, 'BugReport'] = str(str(df.Title.values[0]) + str(df.Description.values[0]))

            elif int(file['top-candidate-scorer-id'][index][-1:]) == 2:
                dataframe.at[index, 'RelevantAnswerDetail'] = df.Candidate_Answer_2.values[0]
                dataframe.at[index, 'BugReport'] = str(str(df.Title.values[0]) + str(df.Description.values[0]))

            elif int(file['top-candidate-scorer-id'][index][-1:]) == 3:
                dataframe.at[index, 'RelevantAnswerDetail'] = df.Candidate_Answer_3.values[0]
                dataframe.at[index, 'BugReport'] = str(str(df.Title.values[0]) + str(df.Description.values[0]))
        filename = str(ids) + ".csv"
        dataframe.to_csv("../../results/bm25/rank_dataframes/"+name+"/"+filename,encoding='utf-8',index=False)

def format_ranks2(name):
    # this function will take the ranks of the top 10 candidates and will add the accepted answer to the dataframe
    goldset_data = pd.read_csv("../../data/bugmentor_gold/"+name+".csv")
    goldsetids = set(goldset_data.IssueID)
    goldsetids = list(goldsetids)
    for ids in goldsetids:
        filename = "../../results/bm25/rank_dataframes/" + name + "/" + str(ids) + ".csv"
        file = pd.read_csv(filename)
        dataframe = pd.DataFrame(file)
        dataframe['AcceptedAnswer'] = ""

        for index, row in dataframe.iterrows():
            gold = goldset_data.loc[goldset_data['IssueID'] == ids]
            dataframe.at[index, 'AcceptedAnswer'] = gold.RelevantAnswerDetail.values[0]

        filename = str(ids) + ".csv"
        dataframe.to_csv("../../results/bm25/rank_dataframes/" + name + "/" + filename, encoding='utf-8', index=False)    

def format_ranks3(name):
    # this function will take the ranks of the top 10 candidates and will add the question to the dataframe
    goldset_data = pd.read_csv("../../data/bugmentor_gold/"+name+".csv")
    goldsetids = set(goldset_data.IssueID)
    goldsetids = list(goldsetids)

    for ids in goldsetids:
        file = pd.read_csv("../../results/bm25/rank_dataframes/" + name + "/" + str(ids) + ".csv")
        dataframe = pd.DataFrame(file)
        dataframe['Question'] = ""

        gold = goldset_data.loc[goldset_data['IssueID'] == ids]
        dataframe['Question'] = gold.Question.values[0]

        filename = str(ids) + ".csv"
        dataframe.to_csv("../../results/bm25/rank_dataframes/" + name + "/" + filename, encoding='utf-8', index=False)

if __name__ == '__main__':
    print("Processing file")
    # filelist = os.listdir("../../data/bugmentor_gold/")
    filelist = ["motivating.csv"]
    
    for file_id in filelist:
        base_filename, extension = os.path.splitext(file_id)
        directory_path = "../../results/bm25/rank_dataframes/"+base_filename
        
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
        
        print("starting with " + base_filename + " -------------------")
        format_ranks(base_filename)
        # format_ranks2(base_filename)
        format_ranks3(base_filename)
        print("finished with " + base_filename + " -------------------")
