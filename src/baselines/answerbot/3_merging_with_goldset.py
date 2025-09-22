import pandas as pd
from autocorrect import Speller
from tqdm import tqdm

spell = Speller(lang='en')

# Function to correct spelling in a given text
def correct_spelling(text):
    words = text.split()
    corrected_words = [spell(word) for word in words]
    return " ".join(corrected_words)


# List of filenames
filenames = ["cpp","java","js","python"]


for name in filenames:
    print("Starting with " + name + " -------------------")
    
    gold = pd.read_csv("./CompleteGoldset/" + name + "_goldset.csv")
    result = pd.read_csv("./CrossProject/ResultSummaries-Top5/summary_res_" + name + ".csv")
    
    merged = pd.concat([gold, result], axis=1)
    merged = merged.drop(columns=['IssueID', 'CandidateAnswer1', 'CandidateAnswer2', 'CandidateAnswer3', 'query'])
    merged = merged.rename(columns={"RelevantAnswerDetail": "AcceptedAnswer", "summary": "AnswerBotSummary"})
    
    # Correct spelling in the relevant columns
    columns_to_check = ['Title', 'Description', 'Question', 'AcceptedAnswer', 'AnswerBotSummary']
    # for column in tqdm(columns_to_check, desc="Spell Checking"):
    #     merged[column] = merged[column].apply(correct_spelling)
    
    merged.to_csv("./CrossProject/AnswerBotResults-Top5/" + name + "_answerbot.csv", index=False)