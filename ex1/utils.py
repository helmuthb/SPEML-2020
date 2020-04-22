import pandas as pd
import recordlinkage
from recordlinkage import preprocessing
from typing import List, Tuple

# Constant: path to the files
FILEPATH = './datasets/DBLP-Scholar/%s.csv'

def read_data(set1: str, set2: str, truth: str, debug: bool=False) \
         -> Tuple[pd.DataFrame, pd.DataFrame, pd.MultiIndex]:
    # suffix for small dataset
    suffix = '_mini' if debug else ''
    # read in data files
    df1 = pd.read_csv(FILEPATH % (set1+suffix))
    df2 = pd.read_csv(FILEPATH % (set2+suffix))
    df_truth = pd.read_csv(FILEPATH % truth)
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
    # MultiIndex of truth values
    mi_truth = pd.MultiIndex.from_tuples(pairs)
    return df1, df2, mi_truth

def read_dataset(name: str, debug: bool=False) -> pd.DataFrame:
    # suffix for small dataset
    suffix = '_mini' if debug else ''
    return pd.read_csv(FILEPATH % (name+suffix))

def clean_attributes(df: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    for n in names:
        df[n+'_clean'] = preprocessing.clean(df[n])
    return df
