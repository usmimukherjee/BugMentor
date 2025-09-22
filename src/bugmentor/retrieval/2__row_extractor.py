import json
import csv
import os

class RowExtractor:
    def __init__(self, bug_id, file_name):
        self.data_file = f"./BM25_scores/{file_name}/{bug_id}.txt"
        self.csv_file = f"../../results/bm25/csv_files/{file_name}/{bug_id}.csv"
    
    def extract_keys(self, data):
        return list(data.keys())
    
    def sanitize(self, values):
        # More robust sanitization may be needed depending on data content
        return values.replace('[', '').replace(']', '').replace('"', '').strip()
    
    def extract_rows(self):
        try:
            with open(self.data_file, 'r') as file:
                data = json.load(file)
                keys = self.extract_keys(data)
                csv_rows = []
                for key in keys:
                    values = str(data[key])
                    sanitized_values = self.sanitize(values)
                    # Create a row as a list of values
                    csv_row = [key] + sanitized_values.split(',')
                    csv_rows.append(csv_row)
                return csv_rows
        except Exception as e:
            print(e)
            return []
    
    def make_csv(self):
        rows = self.extract_rows()
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in rows:
                # Write row as multiple columns
                writer.writerow(row)
        print("CSV creation done for bug ID.")

def main():
    filelist = os.listdir("../../data/bugmentor_gold/")
    for file_id in filelist:
        base_filename, extension = os.path.splitext(file_id)
        if not os.path.exists(f"../../results/bm25/csv_files/{base_filename}"):
            os.makedirs(f"../../results/bm25/csv_files/{base_filename}", exist_ok=True)
        with open("./issuelist/"+base_filename+".txt", 'r') as file:
            for line in file:
                bug_id = int(line.strip())
                extractor = RowExtractor(bug_id, str(base_filename))
                extractor.make_csv()

if __name__ == "__main__":
    main()