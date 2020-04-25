import pandas as pd
import recordlinkage
import random
from recordlinkage import preprocessing
from typing import List, Tuple

# Constant: path to the files
FILEPATH = './datasets/DBLP-Scholar/%s.csv'

def read_data(set1: str, set2: str, truth: str, debug: bool=False) \
         -> Tuple[pd.DataFrame, pd.DataFrame, pd.MultiIndex]:
    # read in data files
    df1 = pd.read_csv(FILEPATH % (set1))
    df2 = pd.read_csv(FILEPATH % (set2))
    df_truth = pd.read_csv(FILEPATH % truth)
    # if debug: only choose approx. 10% of the links
    if debug:
        frac = 0.1
        df_truth = df_truth.sample(frac=frac)
    # memorize indexes used (for subsampling)
    idx1_linked = set()
    idx2_linked = set()
    # process truth dataframe
    pairs = []
    for i in range(len(df_truth.index)):
        # find idx1
        idx1_set = df1[df1.id == df_truth.iloc[i,0]].index.values
        if len(idx1_set) == 0:
            continue
        idx2_set = df2[df2.id == df_truth.iloc[i,1]].index.values
        if len(idx2_set) == 0:
            continue
        # add truth value to pairs
        pairs.append([idx1_set[0], idx2_set[0]])
        # add to used entries
        idx1_linked.add(idx1_set[0])
        idx2_linked.add(idx2_set[0])
    # MultiIndex of truth values
    mi_truth = pd.MultiIndex.from_tuples(pairs)
    # subsample datasets if debug
    if debug:
        # add some of the indexes of df1 and df2
        idx1_linked.update(df1.sample(frac=frac).index.to_list())
        idx2_linked.update(df2.sample(frac=frac).index.to_list())
        # reduce df1 and df2 to the subset
        df1 = df1.iloc[list(idx1_linked)]
        df2 = df2.iloc[list(idx2_linked)]
    return df1, df2, mi_truth

def split_train_test(df1: pd.DataFrame, df2: pd.DataFrame, mi_t: pd.MultiIndex, ratio: float=0.2) \
          -> Tuple[pd.DataFrame, pd.DataFrame, pd.MultiIndex, pd.DataFrame, pd.DataFrame, pd.MultiIndex]:
    # should we go into train?
    to_train = lambda : random.random() < ratio
    # results
    train_set1 = set()
    train_set2 = set()
    train_truth = set()
    test_set1 = set()
    test_set2 = set()
    test_truth = set()
    # split truth, but giving both the first entry
    to_train_truth = lambda: len(train_truth) == 0 or (len(test_truth) != 0 and to_train())
    # go through the truth multiindex
    for m in mi_t:
        d1, d2 = m
        # should it go to train?
        if (d1 in train_set1 and d2 not in test_set2) or (d1 not in test_set1 and d2 in train_set2):
            train_truth.add(m)
            train_set1.add(d1)
            train_set2.add(d2)
        # should it go to test?
        elif (d1 in test_set1 and d2 not in train_set2) or (d1 not in train_set1 and d2 in test_set2):
            test_truth.add(m)
            test_set1.add(d1)
            test_set2.add(d2)
        # should it be dropped?
        elif (d1 in train_set1 and d2 in test_set2) or (d1 in test_set1 and d2 in train_set2):
            pass
        # random split - but first giving each list one entry
        elif to_train_truth():
            train_truth.add(m)
            train_set1.add(d1)
            train_set2.add(d2)
        else:
            test_truth.add(m)
            test_set1.add(d1)
            test_set2.add(d2)
    # assign remaining set1 entries
    for d1, r1 in df1.iterrows():
        if to_train():
            # not yet assigned to the other?
            if d1 not in test_set1:
                train_set1.add(d1)
        else:
            # not yet assigned to the other?
            if d1 not in train_set1:
                test_set1.add(d1)
    for d2, r2 in df2.iterrows():
        if to_train():
            # not yet assigned to the other?
            if d2 not in test_set2:
                train_set2.add(d2)
        else:
            # not yet assigned to the other?
            if d2 not in train_set2:
                test_set2.add(d2)
    # return results
    df1_train = df1.loc[list(train_set1)]
    df2_train = df2.loc[list(train_set2)]
    mi_train = pd.MultiIndex.from_tuples(train_truth)
    df1_test = df1.loc[list(test_set1)]
    df2_test = df2.loc[list(test_set2)]
    mi_test = pd.MultiIndex.from_tuples(test_truth)
    return df1_train, df2_train, mi_train, df1_test, df2_test, mi_test

def read_dataset(name: str, debug: bool=False) -> pd.DataFrame:
    # suffix for small dataset
    suffix = '_mini' if debug else ''
    return pd.read_csv(FILEPATH % (name+suffix))

def clean_attributes(df: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    for n in names:
        df[n+'_clean'] = preprocessing.clean(df[n])
    return df
