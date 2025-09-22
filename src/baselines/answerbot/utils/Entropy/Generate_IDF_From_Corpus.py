#idf_voc.txt

import math
import re
from collections import defaultdict

def compute_idf(file_path):
    # create a defaultdict to store the number of documents containing each term
    doc_freq = defaultdict(int)
    total_docs = 0

    # iterate over each line/document in the file
    with open(file_path, 'r', encoding='utf8', errors='ignore') as file:
        for line in file:
            total_docs += 1
            terms = set(re.findall(r'\b\w+\b|[^\w\s]', line))
            for term in terms:
                if term.isalpha():
                    doc_freq[term] += 1

    # calculate IDF scores for each term
    idf_scores = {}
    for term, freq in doc_freq.items():
        idf_scores[term] = math.log10(total_docs / freq)

    return idf_scores


if __name__ == '__main__':
    idf_scores = compute_idf('..\..\_1_question_retrieval\_1_preprocessing\corpus.txt')

    output_path = 'idf_voc.txt'
    # write the IDF scores to a file
    with open(output_path, 'w', encoding="utf-8") as output_file:
        for term, score in sorted(idf_scores.items()):
            output_file.write(f"{term}   {score}\n")