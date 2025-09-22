import csv
from collections import defaultdict
import os

class CandidateItemSorter:
    def __init__(self, bug_id, file_name):
        self.bug_id = bug_id
        self.csv_file = f"../../results/bm25/csv_files/{file_name}/{bug_id}.csv"
        self.row_map = defaultdict(dict)
        self.candidate_answer_map = {}
    
    def get_csv_file(self, bug_id, file_name):
        return f"../../results/bm25/csv_files/{file_name}/{bug_id}.csv"
    
    def collect_rows(self):
        try:
            with open(self.csv_file, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    self.add_row_to_map(row)
        except Exception as e:
            # do nothing
            pass
    
    def add_row_to_map(self, row):
        if len(row) > 0:
            candidate_qid = int(row[0])
            score_map = {}
            for i in range(1, len(row)):
                score = float(row[i])
                candidate_key = f"{candidate_qid}-{i}"
                score_map[candidate_key] = score
            self.row_map[candidate_qid] = score_map
    
    def construct_candidate_map(self):
        for qid, answer_map in self.row_map.items():
            self.candidate_answer_map.update(answer_map)
    
    def sort_candidates(self):
        sorted_candidates = sorted(self.candidate_answer_map.items(), key=lambda item: item[1], reverse=True)
        return sorted_candidates
    
    def get_top_candidates(self, top_k, name):
        self.collect_rows()
        self.construct_candidate_map()
        sorted_candidates = self.sort_candidates()
        file_path = f"../../results/bm25/ranks/{name}/{self.bug_id}.txt"
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['top-candidate-scorer-id', 'candidate-relevance-score'])
            for i, (key, value) in enumerate(sorted_candidates):
                if i >= top_k:
                    break
                writer.writerow([key, value])

def main():
    print("Processing file")
    filelist = os.listdir("../../data/bugmentor_gold/")
    for file_id in filelist:
        base_filename, extension = os.path.splitext(file_id)
        print("Processing ............")
        if not os.path.exists(f"../../results/bm25/ranks/{base_filename}"):
            os.makedirs(f"../../results/bm25/ranks/{base_filename}", exist_ok=True)
        with open("./issuelist/"+base_filename+".txt", 'r') as file:
            for line in file:
                bug_id = int(line.strip())
                print(bug_id)
                sorter = CandidateItemSorter(bug_id, str(base_filename))
                sorter.get_top_candidates(50, str(base_filename))
        print("END-----------------")
        
if __name__ == "__main__":
    main()