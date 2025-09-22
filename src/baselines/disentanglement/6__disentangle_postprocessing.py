import pandas as pd
import numpy as np
import os

def create_subset(base_filename):
    filelist = os.listdir("../../results/baselines/disentangle/"+base_filename)
    for name in filelist:
        df = pd.read_csv("../../results/baselines/disentangle/"+name)
        # get a subset of the data of 100 rows if exists else take the whole data
        if len(df) > 100:
            df = df.head(100)
        df.to_csv("../../results/baselines/disentangle100/"+base_filename+"/"+name, index=False)


if __name__ == '__main__':
    # getting directory names
    filelist = os.listdir("../../results/baselines/disentangle")
    for name in filelist:
        if name == 'readme.md' :
            continue
        base_filename, extension = os.path.splitext(name)
        directory_path = "../../results/baselines/disentangle100/"+base_filename
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
        print("\nstarting with " + base_filename + " -------------------")
        create_subset(base_filename)
        print("\nfinished with " + base_filename + " -------------------")