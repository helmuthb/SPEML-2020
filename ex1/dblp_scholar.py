# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # DBLP Scholar: Record Linkage attacks
# 
# This Notebook implements a few sample record linkage attacks and calculates the accuracy, precision and recall.

import recordlinkage
import time
from recordlinkage import compare
# import utility functions for dealing with datasets
from utils import read_data, clean_attributes

# set debug flag:
debug = False

# read DBLP and Google Scholar dataset
dataDBLP, dataScholar, links = read_data(
    'DBLP1', 'Scholar', 'DBLP-Scholar_perfectMapping', debug)

if debug:
    display(dataDBLP)
    display(dataScholar)
    display(links)

# ## 2. Cleaning and Pre-Processing

# cleaning: bring all to lowercase, remove unwanted tokens
dataDBLP = clean_attributes(dataDBLP, ['title', 'authors', 'venue'])
dataScholar = clean_attributes(dataScholar, ['title', 'authors', 'venue'])
# show the dataframes
if debug:
    display(dataDBLP)
    display(dataScholar)

# %% 

def print_experiment_evaluation(matches):
    precision = recordlinkage.precision(links, matches)
    recall = recordlinkage.recall(links, matches)
    fscore = recordlinkage.fscore(links, matches)
    print(f"--> Precision: {precision}")
    print(f"--> Recall: {recall}")
    print(f"--> F-score: {fscore}")
    display(recordlinkage.confusion_matrix(links, matches))


def run_experiment(indexer_variant, comparison_variant, classification_variant):
    if indexer_variant == 0:   
        indexer = recordlinkage.index.SortedNeighbourhood('year')
        pairs = indexer.index(dataDBLP, dataScholar)
        if debug:
            print(f"Number of candidates (sortedneighbour window=3):\n{len(pairs)}")
    elif indexer_variant == 1:
        indexer = recordlinkage.index.SortedNeighbourhood('year', window=1)
        pairs = indexer.index(dataDBLP, dataScholar)
        if debug:
            print(f"Number of candidates (sortedneighbour window=1)):\n{len(pairs)}")
    else:
        print("unknown indexer variant %d" % indexer_variant)
        return

    comp = recordlinkage.Compare()
    if comparison_variant == 0:
        print("standard string comparison ")
        comp.add(compare.String('title_clean', 'title_clean'))
        comp.add(compare.String('authors_clean', 'authors_clean'))
        comp.add(compare.String('venue_clean', 'venue_clean'))
    elif comparison_variant == 1:
        print("string comparison: title jaro, authors levenshtein, venue jaro")
        comp.add(compare.String('title_clean', 'title_clean', method='jaro'))
        comp.add(compare.String('authors_clean', 'authors_clean'))
        comp.add(compare.String('venue_clean', 'venue_clean', method='jaro'))
    else:
        print("unknown comparison variant %d", comparison_variant)
        return 

    start = time.time()
    result = comp.compute(pairs, dataDBLP, dataScholar)
    print("comparing took: %.2fs" % (time.time() - start))

    variants = 3
    for i in range(variants):
        classification_variant = i
        if classification_variant == 0:
            print("simple classifier")
            # simple classifier: add the values and use a threshold of 2
            matches = result[result[0]+result[1]+result[2]>2].index
        elif classification_variant == 1:
            print("logistic regression classifier")
            classifier = recordlinkage.LogisticRegressionClassifier()
            matches = classifier.fit_predict(result, links)
        elif classification_variant == 2:
            print("svm classifier")
            classifier = recordlinkage.SVMClassifier()
            matches = classifier.fit_predict(result, links)

        if debug:
            print("%d matches" % len(matches))
            print_experiment_evaluation(matches)
            # display(matches)

    return matches

indexer_variants = 2
comparison_variants = 2

# run_experiment(1, 1, 0)

matches = []
for indexer in range(indexer_variants):
    for comparer in range(comparison_variants):
        matches.append(run_experiment(indexer, comparer, 0))


# %% [markdown]
# ## 6. Evaluation
# 
# We use again the recordlinkage package for calculating evaluation values of the results.

# %%
for match in matches:
    precision = recordlinkage.precision(links, match)
    recall = recordlinkage.recall(links, match)
    fscore = recordlinkage.fscore(links, match)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-score: {fscore}")
    display(recordlinkage.confusion_matrix(links, match))



# %%
