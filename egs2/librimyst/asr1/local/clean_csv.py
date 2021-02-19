import os
import pandas as pd
import argparse
import csv

def clean_myst(df):
    # remove Nans
    df = df[df['transcript'].apply(lambda x: type(x) == str)]
    # filter out samples with characters that are not from english alphabet e.g. speical symbols, digits etc.
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.' "
    df_filtered = df[df['transcript'].apply(lambda x: all(c in alphabet for c in x))]
    # remove transcripts with less 10 characters long
    df_filtered = df_filtered[df_filtered['transcript'].apply(lambda x: len(x) >= 10)]
    return df_filtered

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv_fname', '-C', help="path to the dataset csv")
    args = parser.parse_args()
    csv_fname = args.csv_fname
    df = pd.read_csv(args.csv_fname)
    df_clean = clean_myst(df)
    df_clean.to_csv(os.path.splitext(csv_fname)[0] + "_clean.csv", index=False)
    print('done')
