import os
import pandas as pd
import argparse
import csv
import bisect

def train_dev_test_split(df, train_dev_test_split=[80,10,10], seed=1234):
    groups = df.groupby('reader_id')
    cumsums = groups.count().sample(frac=1, random_state=seed).cumsum()
    
    train_num = train_dev_test_split[0] * len(df) // 100
    train_pos = bisect.bisect_left(list(cumsums['fname']), train_num)
    train_readers = cumsums[:train_pos]
    train_readers_idxs = list(train_readers.index)
    train_set = df[df['reader_id'].isin(train_readers_idxs)].reset_index(drop=True)
    print(f"number of users in train set = {len(train_readers_idxs)}. Number of samples in train set = {len(train_set)}.")

    dev_num = train_dev_test_split[1] * len(df) // 100 + train_num
    dev_pos = bisect.bisect_left(list(cumsums['fname']), dev_num)
    dev_readers = cumsums[train_pos:dev_pos]
    dev_readers_idxs = list(dev_readers.index)
    dev_set = df[df['reader_id'].isin(dev_readers_idxs)].reset_index(drop=True)
    print(f"number of users in dev set = {len(dev_readers_idxs)}. Number of samples in dev set = {len(dev_set)}.")

    test_readers = cumsums[dev_pos:]
    test_readers_idxs = list(test_readers.index)
    test_set = df[df['reader_id'].isin(test_readers_idxs)].reset_index(drop=True)
    print(f"number of users in test set = {len(test_readers_idxs)}. Number of samples in test set = {len(test_set)}.")

    return train_set, dev_set, test_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv_fname', '-C', help="path to the dataset csv")
    parser.add_argument('-proportions', '-P', nargs=3, type=int, help="percentage of train, dev, test splits (3 numbers required)", default=[80, 10, 10])
    parser.add_argument('-seed', '-S', type=int, help="percentage of train, dev, test splits (3 numbers required)", default=1234)

    args = parser.parse_args()

    df = pd.read_csv(args.csv_fname)
    train_df, dev_df, test_df = train_dev_test_split(df, args.proportions, args.seed)
    
    csv_base_name = os.path.splitext(args.csv_fname)[0]
    train_df.to_csv(csv_base_name + "_train.csv", index=False)
    dev_df.to_csv(csv_base_name + "_dev.csv", index=False)
    test_df.to_csv(csv_base_name + "_test.csv", index=False)
    print('done')
