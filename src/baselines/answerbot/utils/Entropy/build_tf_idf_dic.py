from __future__ import absolute_import
from __future__ import print_function
from utils.file_util import write_file
import pandas as pd
path_of_voc = 'E:\Pycharm Projects\Thesis - QA\AnswerBot\CrossProject\idf_vocab\\'


# def read_voc(filename):
#     with open(path_of_voc+ filename +'.csv', 'r', encoding='utf8', errors='ignore') as file:
#         # file = open(path_of_voc)
#         voc = {}
#         for line in file:
#             print(line)
#             word_idf = line.split('   ')
#             word = word_idf[0]
#             idf = float(word_idf[1].strip())
#             voc[word] = idf
#     return voc

def read_voc(filename):
    file_path = path_of_voc + 'complete.csv'
    df = pd.read_csv(file_path, engine='python')
    voc = df.set_index('word')['idf'].to_dict()
    return voc
