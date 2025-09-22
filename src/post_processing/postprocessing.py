import pandas as pd
import warnings
import os
import numpy as np

# ignoring warnings
warnings.filterwarnings('ignore')

# Ranklist Formatting
def format_ranks(name):
    goldset_data = pd.read_csv("../../data/bugmentor_gold/" + name + ".csv")
    goldsetids = set(goldset_data.IssueID)
    goldsetids = list(goldsetids)

    for ids in goldsetids:

        filename = "../../results/bm25_mistral/" + name + "/" + str(ids) + ".csv"
        file = pd.read_csv(filename)

        dataframe = pd.DataFrame(file)
        dataframe['GoldBugReport'] = ""

        for index, row in dataframe.iterrows():
            gold = goldset_data.loc[goldset_data['IssueID'] == ids]
            dataframe.at[index, 'Question'] = gold.Question.values[0]
            dataframe.at[index, 'GoldBugReport'] = gold.Title.values[0] + gold.Description.values[0]

        directory_path = "../../results/ablation_bm25mistral/" + name 
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
        dataframe.to_csv(directory_path+"/"+ str(ids) + ".csv", index=False)

if __name__ == '__main__':
    filelist = os.listdir("../../results/bm25_mistral/")
    for name in filelist:
            print("starting with " + name + " -------------------")
            format_ranks(name)
            print("\n finished with " + name + " -------------------")

