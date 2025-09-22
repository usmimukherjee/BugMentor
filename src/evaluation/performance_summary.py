import os
import pandas as pd
from collections import defaultdict

BASE_DIR = "../../evaluated/bugmentor/"
OUTPUT_DIR = "../../performance/bugmentor/"

def get_unique_top_n(scores, n):
    """
    Extracts top n unique values from a set of scores.
    """
    unique_scores = sorted(scores, reverse=True)
    if len(unique_scores) > n:
        return unique_scores[:n]
    return unique_scores

def process_files_for_name(name):
    """
    Processes all files in a directory corresponding to the given name,
    computing top unique values for each metric and saving the results to a CSV.
    """
    path = os.path.join(BASE_DIR, name)
    files = os.listdir(path)

    # Use a set to automatically handle uniqueness
    all_scores = defaultdict(set)

    for filename in files:
        filepath = os.path.join(path, filename)
        dataframe = pd.read_csv(filepath)
        for metric in ['BLEU_SCORE', 'ROUGE_SCORE', 'METEOR_SCORE', 'SEMANTIC_SIMILARITY']:
            all_scores[metric].update(dataframe[metric].dropna())

    # Prepare result DataFrame using list comprehensions for each metric
    result_df = pd.DataFrame({
        'BLEU_MAX': [max(get_unique_top_n(all_scores['BLEU_SCORE'], n)) if all_scores['BLEU_SCORE'] else float('nan') for n in [1, 3, 5]],
        'ROUGE_MAX': [max(get_unique_top_n(all_scores['ROUGE_SCORE'], n)) if all_scores['ROUGE_SCORE'] else float('nan') for n in [1, 3, 5]],
        'METEOR_MAX': [max(get_unique_top_n(all_scores['METEOR_SCORE'], n)) if all_scores['METEOR_SCORE'] else float('nan') for n in [1, 3, 5]],
        'SBERT_MAX': [max(get_unique_top_n(all_scores['SEMANTIC_SIMILARITY'], n)) if all_scores['SEMANTIC_SIMILARITY'] else float('nan') for n in [1, 3, 5]]
    })

    # Save results to CSV
    output_file = os.path.join(OUTPUT_DIR, f"{name}.csv")
    result_df.to_csv(output_file, encoding='utf-8', index=False)

def main():
    filelist = os.listdir(BASE_DIR)
    for name in filelist:
        if name in ['readme']:
            continue
        print(f"Processing {name}...")
        process_files_for_name(name)
        print(f"Done {name}")

if __name__ == '__main__':
    main()