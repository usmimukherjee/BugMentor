
import pandas as pd
import csv
import math
import operator
import os
import sys
import numpy as np
import copy
import time
import warnings
warnings.filterwarnings("ignore") 
from nltk import word_tokenize
from six.moves import range
from gensim.models import Word2Vec, word2vec
from nltk import word_tokenize
from gensim.models import KeyedVectors
from gensim import models
from utils.Entropy.build_tf_idf_dic import read_voc
from utils.Entity.Entity_Analysis import get_entities_from_word_list, get_entity_score
from utils.str_util import split_into_paragraph, split_into_sentence, unicode2str
from utils.preprocessing_util import preprocessing_for_ans_sent
from utils.Order.Order_Analysis import get_order_score
from utils.Pattern.Pattern_Analysis import get_pattern_score
from utils.HTMLTag.HTML_Analysis import get_html_score
from utils.Entropy.Entropy_Analysis import get_entropy_score

global path_of_stopwords_EN
global w2v_model_fpath
global vocab_fpath
path_of_stopwords_EN = "AnswerBot/utils/StopWords_EN.txt"
w2v_model_fpath=os.path.join("AnswerBot/SO_vectors_200.bin")


def read_EN_stopwords():
    sw_set = set()
    f = open(path_of_stopwords_EN)
    for line in f:
        sw_set.add(line.strip())
    return sw_set


def remove_stopwords(sent, sw):
    if type(sent) is str:
        wlist = word_tokenize(sent)
    elif type(sent) is list:
        wlist = sent
    else:
        raise Exception("Wrong type for removing stopwords!")
    sent_words = []
    for w in wlist:
        if w == '':
            continue
        if w not in sw:
            sent_words.append(w)
    return sent_words


def init_doc_matrix(doc,w2v):

    matrix = np.zeros((len(doc),200)) #word embedding size is 100
    for i, word in enumerate(doc):
        if word in w2v.key_to_index:
            matrix[i] = np.array(w2v[word])
        # if word in w2v.wv.vocab:
        #     matrix[i] = np.array(w2v.wv[word])

    #l2 normalize
    try:
        norm = np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
        matrix = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm!=0)
        #matrix = matrix / np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
    except RuntimeWarning:
        print(doc)

    #matrix = np.array(preprocessing.normalize(matrix, norm='l2'))

    return matrix

def init_doc_idf_vector(doc,idf):
    idf_vector = np.zeros((1,len(doc)))  # word embedding size is 200
    for i, word in enumerate(doc):
        if word in idf:
            idf_vector[0][i] = idf[word]

    return idf_vector


def load_w2v_model():
    word2vector_model = KeyedVectors.load_word2vec_format(w2v_model_fpath, binary=True)
    return word2vector_model


def load_idf_vocab():
    import csv
    vocab_dict = dict()
    try:
        with open(vocab_fpath, 'r',encoding="utf-8") as csvfile:
            rd = csv.reader(csvfile)
            next(rd)
            for row in rd:
                vocab_dict[str(row[0])] = float(row[1])
    except Exception as e:
        print(f"UnicodeDecodeError: {e}")
    return vocab_dict


def preprocessing_for_query(q):
    # basic preprocessing for query
    qw = word_tokenize(q[0].lower())
    stopwords = read_EN_stopwords()
    qw = remove_stopwords(qw, stopwords)
    return qw

def sim_doc_pair(matrix1,matrix2,idf1,idf2):
    sim12 = (idf1*(matrix1.dot(matrix2.T).max(axis=1))).sum() / idf1.sum()

    sim21 = (idf2*(matrix2.dot(matrix1.T).max(axis=1))).sum() / idf2.sum()

    return (sim12 + sim21) / 2.0
    # total_len = matrix1.shape[0] + matrix2.shape[0]
    # return sim12 * matrix2.shape[0] / total_len + sim21 * matrix1.shape[0] / total_len

def calc_wordvec_similarity(vec1, vec2):
    vec1 = vec1.reshape(1, len(vec1))
    vec2 = vec2.reshape(1, len(vec2))
    x1_norm = np.sqrt(np.sum(vec1 ** 2, axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(vec2 ** 2, axis=1, keepdims=True))
    prod = np.sum(vec1 * vec2, axis=1, keepdims=True)
    cosine_sim = prod / (x1_norm * x2_norm)
    return cosine_sim[0][0]

def calc_similarity(word_list_1, word_list_2, idf_voc, word2vector_model):
    if len(word_list_1) == 0 or len(word_list_2) == 0:
        return 0.0

    sim_up = 0
    sim_down = 0
    for w1 in word_list_1:
        w1_unicode = w1.encode('utf-8')
        if w1_unicode in word2vector_model.key_to_index:
            w1_vec = word2vector_model[w1_unicode]
            # maxsim
            maxsim = 0.0
            for w2 in word_list_2:
                # word similarity
                w2_unicode = w2.decode('utf-8')
                if w2_unicode in word2vector_model:
                    w2_vec = word2vector_model[w2_unicode]
                    sim_tmp = calc_wordvec_similarity(w1_vec, w2_vec)
                    if sim_tmp > maxsim:
                        maxsim = sim_tmp
            # if exist in idf
            if w1 in idf_voc:
                idf = idf_voc[w1]
                sim_up += maxsim * idf
                sim_down += idf
            # else:
            # print("%s not in idf vocabulary!" % w1)
    if sim_down == 0:
        # print(("sim_down = 0!\n word sent 1 %s\nword sent 2 %s" % (word_list_1, word_list_2)))
        return 0
    return sim_up / sim_down


def get_dq(query_w, topnum, questions, query_idf, query_matrix):
    rank = []
    #stopwords = read_EN_stopwords()
    cnt = 0
    for index, row in questions.iterrows():
        if type(row.matrix) is float:
            continue
        sim = sim_doc_pair(query_matrix,row.matrix, query_idf, row.idf_vector)
        rank.append([row.IssueID, sim])
        cnt += 1
        if cnt % 10000 == 0:
            print(("Processed %s questions...%s" % (cnt, time.strftime('%Y-%m-%d %H:%M:%S'))))

    # format: [id,sim]
    rank.sort(key=operator.itemgetter(1), reverse=True)
    # top_dq,rank
    top_dq = []
    for i in range(0, len(rank), 1):
        id = rank[i][0]
        sim = rank[i][1]
        rank.append(id)
        if i < topnum:
            qs = questions[questions['IssueID'] == id].iloc[0].IssueID
            top_dq.append((qs, sim))
    return top_dq


def write_list_to_csv(list_tmp, csv_fpath, header):
    with open(csv_fpath, 'w', newline='', encoding="utf-8") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(header)
        for x in list_tmp:
            try:
                if(isinstance(x, int)):
                    x=[x]
                wr.writerow(x)
            except Exception as e:
                print(("Error %s" % e))
    print(("Write %s successfully!" % csv_fpath))
    
    
def preprocess_all_questions(questions, idf, w2v, stopword):
    processed_questions = list()
    stopwords = stopword
    # iterating through the questions dataframe
    for index, row in questions.iterrows():
        question = row['Question']
        if type(question) is str:
            title_words = remove_stopwords(question, stopwords)
            if len(title_words) <= 2:
                continue
            if title_words[-1] == '?':
                title_words = title_words[:-1]
            
            questions.loc[index, 'title'] = ""
            questions.loc[index, 'matrix'] = ""
            questions.loc[index, 'idf_vector'] = ""

            # Assign values to the columns
            questions.at[index, 'title'] = title_words
            questions.at[index, 'matrix'] = init_doc_matrix(title_words, w2v)
            questions.at[index, 'idf_vector'] = init_doc_idf_vector(title_words, idf)
            processed_questions.append(question)
        else:
            continue
    return processed_questions, questions

def read_correspond_answers_from_table(top_dq_id_and_sim,filename):
    # read files from corpus dataframe
    answers = pd.read_csv('AnswerBot/Corpus/'+filename+'_corpus.csv')
    answers_list = []
    for (q_id, sim) in top_dq_id_and_sim:
        answers_list.append(answers[answers['IssueID'] == q_id].iloc[0])
    return answers_list
    
def get_ss(query_word, top_relevant_paragraph_num, top_dq_id_and_sim, stopword,filename):
    sent_list = []
    # Relevance
    max_Relevance = -sys.float_info.max
    min_Relevance = sys.float_info.max
    # Entity
    max_Entity = -sys.float_info.max
    min_Entity = sys.float_info.max
    # Score
    # Score
    max_A_Score = -sys.float_info.max
    min_A_Score = sys.float_info.max
    # Order
    max_Order = -sys.float_info.max
    min_Order = sys.float_info.max
    # Pattern
    max_Pattern = -sys.float_info.max
    min_Pattern = sys.float_info.max
    # HTMLTag
    max_HTMLTag = -sys.float_info.max
    min_HTMLTag = sys.float_info.max
    # Entropy
    max_Entropy = -sys.float_info.max
    min_Entropy = sys.float_info.max
    # Score
    max_Score = -sys.float_info.max
    min_Score = sys.float_info.max

    # For Entropy calculation : preprocessing for query
    query_words = query_word
        
    # load stopwords
    stopwords = stopword
    
    # load idf voc
    idf_voc = read_voc(filename)
    # question-level


    answers_list = read_correspond_answers_from_table(top_dq_id_and_sim,filename)

    for (q_id, sim) in top_dq_id_and_sim:
        # Get the list of Issue IDs in the data
        issue_ids = [issue['IssueID'] for issue in answers_list]
        
        # Initialize an empty list to store matching candidate answers
        answers = []

        # Iterate through the data to find candidate answers with the target Issue ID
        for issue_data in answers_list:
            if issue_data['IssueID'] == q_id:
                # Check if the 'Candidate_Answer_1' key exists in the dictionary
                if not pd.isna(issue_data['Candidate_Answer_1']):
                    answers.append({
                        'id': 1,
                        'Body': issue_data['Candidate_Answer_1']
                    })
                # Check if the 'Candidate_Answer_2' key exists in the dictionary
                if not pd.isna(issue_data['Candidate_Answer_2']):
                    answers.append({
                        'id': 2,
                        'Body': issue_data['Candidate_Answer_2']
                    })
                # Check if the 'Candidate_Answer_3' key exists in the dictionary
                if not pd.isna(issue_data['Candidate_Answer_3']):
                    answers.append({
                        'id': 3,
                        'Body': issue_data['Candidate_Answer_3']
                    })

        # Iterate through the list of matching candidate answers
        for answer_tmp in answers:
            
            sentences = answer_tmp['Body']
            order = 1
            a_id = answer_tmp['id']
            # paragraph-level
            

            clean_sent = preprocessing_for_ans_sent(sentences)
            
            # relevance
            relevance = sim
            max_Relevance = relevance if max_Relevance < relevance else max_Relevance
            min_Relevance = relevance if min_Relevance > relevance else min_Relevance
                            
            # # score
            # a_score = answer_tmp.score
            # max_A_Score = a_score if max_A_Score < a_score else max_A_Score
            # min_A_Score = a_score if min_A_Score > a_score else min_A_Score
            
            # Order
            order = get_order_score(order)
            order += 1
            max_Order = order if max_Order < order else max_Order
            min_Order = order if min_Order > order else min_Order
            # Pattern
            pattern = get_pattern_score(clean_sent)
            max_Pattern = pattern if max_Pattern < pattern else max_Pattern
            min_Pattern = pattern if min_Pattern > pattern else min_Pattern
            # HTML Tag
            htmltag = get_html_score(sentences)
            max_HTMLTag = htmltag if max_HTMLTag < htmltag else max_HTMLTag
            min_HTMLTag = htmltag if min_HTMLTag > htmltag else min_HTMLTag
            # Entropy
            entropy = get_entropy_score(query_words, clean_sent, stopwords, idf_voc)
            max_Entropy = entropy if max_Entropy < entropy else max_Entropy
            min_Entropy = entropy if min_Entropy > entropy else min_Entropy
            sent_list.append([clean_sent, relevance, order, pattern, htmltag, entropy, a_id, q_id])


    # Normalization from 1.0 -> 2.0 except Score
    Normalized_sent_list_tmp = []
    for [clean_sent, relevance, order, pattern, htmltag, entropy, a_id, q_id] in sent_list:
        relevance = 1.0 if (max_Relevance - min_Relevance) == 0 else 1.0 + (relevance - min_Relevance) / (
                max_Relevance - min_Relevance)
        # entity = 1.0 if (max_Entity - min_Entity) == 0 else 1.0 + (entity - min_Entity) / (max_Entity - min_Entity)
        # a_score = 1.0 if (max_A_Score - min_A_Score) == 0 else 1.0 + (a_score - min_A_Score) / (
        #         max_A_Score - min_A_Score)
        order = 1.0 if (max_Order - min_Order) == 0 else 1.0 + (order - min_Order) / (max_Order - min_Order)
        pattern = 1.0 if (max_Pattern - min_Pattern) == 0 else 1.0 + (pattern - min_Pattern) / (
                max_Pattern - min_Pattern)
        htmltag = 1.0 if (max_HTMLTag - min_HTMLTag) == 0 else 1.0 + (htmltag - min_HTMLTag) / (
                max_HTMLTag - min_HTMLTag)
        entropy = 1.0 if (max_Entropy - min_Entropy) == 0 else 1.0 + (entropy - min_Entropy) / (
                max_Entropy - min_Entropy)
        Score = relevance * order * pattern * htmltag * entropy
        max_Score = Score if max_Score < Score else max_Score
        min_Score = Score if min_Score > Score else min_Score
        Normalized_sent_list_tmp.append([clean_sent, Score, a_id, q_id])
    del sent_list

    # Normalization Score from 0.0 -> 1.0
    Normalized_sent_list = []
    for [clean_sent, Score, a_id, q_id] in Normalized_sent_list_tmp:
        Score = (Score - min_Score) / (max_Score - min_Score)
        Normalized_sent_list.append([clean_sent, Score, a_id, q_id])
    del Normalized_sent_list_tmp

    # sort by Score then q_id
    Normalized_sent_list.sort(key=operator.itemgetter(1, 3), reverse=True)
    return [x[0] for x in Normalized_sent_list[:top_relevant_paragraph_num]]


def load_qs_result(rq_fpath):
    import pandas as pd
    rq_res = []
    df = pd.read_csv(rq_fpath)
    for idx, row in df.iterrows():
        rq_res.append([row[0], eval(row[1])])
    return rq_res

def load_ss_result(ss_fpath):
    import pandas as pd
    ss_res = list()
    df = pd.read_csv(ss_fpath)
    for idx, row in df.iterrows():
        ss_res.append((row[0], eval(row[1])))
    return ss_res

def get_summary(query, top_ss, topk):
    selected_sentence = MMR_Analysis(query, top_ss, topk)
    summary = '\n'.join([x for x in selected_sentence])
    return summary


def MMR_Analysis(query, top_ss, topnum):
    # print 'MMR analysis', time.strftime('%Y-%m-%d %H:%M:%S')
    sim_matrix = build_sim_matrix(query, top_ss)
    rank_list = MMR_Algorithm(sim_matrix, topnum)
    selected_sentence = []
    for rank in rank_list:
        if 0 <= rank < len(top_ss):
            selected_sentence.append(top_ss[rank])
    return selected_sentence




def build_sim_matrix(query, top_ss):
    # add query
    top_ss_tmp = copy.deepcopy(top_ss)
    top_ss_tmp.append(query)
    len_of_paragraph = len(top_ss_tmp)
    sim_matrix = [[0 for col in range(len_of_paragraph)] for row in range(len_of_paragraph)]
    # doc sim parameter
    w2v_model = load_w2v_model()
    stopwords = read_EN_stopwords()
    idf_voc = load_idf_vocab()

    # tokenize
    for i in range(len(top_ss_tmp)):
        top_ss_tmp[i] = word_tokenize(top_ss_tmp[i])
        top_ss_tmp[i] = remove_stopwords(top_ss_tmp[i], stopwords)

    for i in range(0, len_of_paragraph, 1):
        for j in range(0, i + 1, 1):
            if i == j:
                sim_matrix[i][j] = 1.0
            else:
                sim_matrix[i][j] = (calc_similarity(top_ss_tmp[i], top_ss_tmp[j], idf_voc, w2v_model) +
                                    calc_similarity(top_ss_tmp[j], top_ss_tmp[i], idf_voc, w2v_model)) / 2.0
                sim_matrix[j][i] = sim_matrix[i][j]
    return sim_matrix


def MMR_Algorithm(sim_matrix, topnum):
    iteration = 1
    query_index = len(sim_matrix) - 1
    # init
    Set = []
    Rest = [i for i in range(0, query_index - 1, 1)]
    # parameter
    Lambda = 0.5
    while iteration <= topnum:
        # find most sim with query
        most_sim_with_query_index = -1
        max_dq_sim = -1
        for i in range(0, query_index, 1):
            if sim_matrix[i][query_index] > max_dq_sim:
                max_dq_sim = sim_matrix[i][query_index]
                most_sim_with_query_index = i
        if len(Set) == 0 and most_sim_with_query_index in Rest:
            Set.append(most_sim_with_query_index)
            Rest.remove(most_sim_with_query_index)
        else:
            max_MMR = -sys.float_info.max
            max_MMR_idx = -1
            for cur in Rest:
                max_dd_sim = -sys.float_info.max
                max_dd_idx = -1
                for i in Set:
                    if sim_matrix[cur][i] > max_dd_sim:
                        max_dd_sim = sim_matrix[cur][i]
                        max_dd_idx = i
                MRR_tmp = Lambda * sim_matrix[cur][query_index] - (1 - Lambda) * sim_matrix[cur][max_dd_idx]
                if MRR_tmp > max_MMR:
                    max_MMR = MRR_tmp
                    max_MMR_idx = cur
            Set.append(max_MMR_idx)
            if max_MMR_idx in Rest:
                Rest.remove(max_MMR_idx)
        iteration += 1
    return Set




if __name__ == '__main__':

    filenames = ["cpp","java","js","python"]
    
    for name in filenames:
        print("processing file: ", name)
        topnum = 10
        
        vocab_fpath=os.path.join("./idf_vocab/"+name+".csv")
        
        
        # test query
        df = pd.read_csv("../Data/" + name + "_goldset.csv")
        query_list = df[['Question']].values.tolist()
        w2v_model = load_w2v_model()
        idf_vocab = load_idf_vocab()
        dq_res = list()
        stopword = read_EN_stopwords()
        
        # corpus
        repo = pd.read_csv('../Data/'+filename+'_corpus.csv')
        repo_qa = repo[['Question']]
        
        global questions_repo
        questions, questions_repo = preprocess_all_questions(repo, idf_vocab, w2v_model, stopword)

        for query in query_list:
            # preprocessing the query,getting doc matrix and idf
            query_word = preprocessing_for_query(query)
            query_matrix = init_doc_matrix(query_word, w2v_model)
            query_idf = init_doc_idf_vector(query_word , idf_vocab)
            
            # getting test query against corpus queries
            top_dq = get_dq(query_word, topnum, questions_repo, query_idf, query_matrix)
            cur_res_dict = []
            for i in range(len(top_dq)):
                q = top_dq[i][0]
                sim = top_dq[i][1]
                cur_res_dict.append((q, round(sim, 2)))
            dq_res.append([query, cur_res_dict])
        
        # res_dir = 'E:\Pycharm Projects\Thesis - QA\AnswerBot\CrossProject\Results'
        # dqres_fpath = os.path.join(res_dir, 'rq_res_'+name+'.csv')
        # header = ["query", "rq_id_list"]
        # write_list_to_csv(dq_res, dqres_fpath, header)

        print('sentence selection...', time.strftime('%Y-%m-%d %H:%M:%S'))
        ss_res = list()
        for query, top_dq_id_and_sim in dq_res:
            top_ss = get_ss(query_word, topnum, top_dq_id_and_sim, stopword,name)
            ss_res.append((query, top_ss))
  
        # done till here - continiue from here
        print('get summary...', time.strftime('%Y-%m-%d %H:%M:%S'))
        res = list()
        for query, ss in ss_res:
            query = ' '.join(preprocessing_for_query(query))
            sum = get_summary(query, ss, 5)
            res.append([query, sum])
        
        res_dir = './Results'
        res_fpath = os.path.join(res_dir, 'summary_res_'+name+'.csv')
        header = ["query", "summary"]
        write_list_to_csv(res, res_fpath, header)