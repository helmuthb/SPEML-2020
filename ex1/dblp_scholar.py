# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # DBLP Scholar: Record Linkage attacks
# 
# This Notebook implements a few sample record linkage attacks and calculates the accuracy, precision and recall.

import recordlinkage
import time
import matplotlib.pyplot as plt
from recordlinkage import compare
# import utility functions for dealing with datasets
from utils import read_data, clean_attributes, split_train_test

# set debug flag:
debug = False

# %%

# read DBLP and Google Scholar dataset
dataDBLP, dataScholar, links = read_data(
    'DBLP1', 'Scholar', 'DBLP-Scholar_perfectMapping', debug)

if debug and "display" in dir():
    display(dataDBLP)
    display(dataScholar)
    display(links)

# ## 2. Cleaning and Pre-Processing

# cleaning: bring all to lowercase, remove unwanted tokens
dataDBLP = clean_attributes(dataDBLP, ['title', 'authors', 'venue'])
dataScholar = clean_attributes(dataScholar, ['title', 'authors', 'venue'])
# show the dataframes
if debug and "display" in dir():
    display(dataDBLP)
    display(dataScholar)

#%%

# Split into train and test dataset
dataDBLP_train, dataScholar_train, links_train, \
    dataDBLP_test, dataScholar_test, links_test = split_train_test(
        dataDBLP, dataScholar, links)
if debug:
    print(f"Sizes of train set: {len(dataDBLP_train)}, {len(dataScholar_train)}, {len(links_train)}")
    print(f"Sizes of test set: {len(dataDBLP_test)}, {len(dataScholar_test)}, {len(links_test)}")
# %% 

def print_experiment_evaluation(matches):
    precision = 0
    recall = 0
    fscore = 0

    if len(matches) > 0:
        precision = recordlinkage.precision(links_test, matches)
        recall = recordlinkage.recall(links_test, matches)
        fscore = recordlinkage.fscore(links_test, matches) if recall+precision>0 else 0
        
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-score: {fscore}")
    print(recordlinkage.confusion_matrix(links_test, matches))

    return precision, recall, fscore


def run_experiment(indexer_variant, comparison_variant, classification_variant):
    config_description = ""

    if indexer_variant == 0:   
        config_description += "n3"
        indexer = recordlinkage.index.SortedNeighbourhood('year')
        pairs_train = indexer.index(dataDBLP_train, dataScholar_train)
        pairs_test = indexer.index(dataDBLP_test, dataScholar_test)
        if debug:
            print("Number of candidates (sortedneighbour window=3):")
            print(f"{len(pairs_train)} (train), {len(pairs_test)} (test)")
    elif indexer_variant == 1:
        config_description += "n1"
        indexer = recordlinkage.index.SortedNeighbourhood('year', window=1)
        pairs_train = indexer.index(dataDBLP_train, dataScholar_train)
        pairs_test = indexer.index(dataDBLP_test, dataScholar_test)
        if debug:
            print("Number of candidates (sortedneighbour window=1)):")
            print(f"{len(pairs_train)} (train), {len(pairs_test)} (test)")
    else:
        print("unknown indexer variant %d" % indexer_variant)
        return

    comp = recordlinkage.Compare()
    if comparison_variant == 0:
        config_description += "Default"
        print("standard string comparison ")
        comp.add(compare.String('title_clean', 'title_clean'))
        comp.add(compare.String('authors_clean', 'authors_clean'))
        comp.add(compare.String('venue_clean', 'venue_clean'))
    elif comparison_variant == 1:
        config_description += "Jaro"
        print("string comparison: title jaro, authors levenshtein, venue jaro")
        comp.add(compare.String('title_clean', 'title_clean', method='jaro'))
        comp.add(compare.String('authors_clean', 'authors_clean'))
        comp.add(compare.String('venue_clean', 'venue_clean', method='jaro'))
    else:
        print("unknown comparison variant %d", comparison_variant)
        return 

    print("Start compare for training data set")
    start = time.time()
    result_train = comp.compute(pairs_train, dataDBLP_train, dataScholar_train)
    print("Compare on training data took %.2fs" % (time.time() - start))
    print("Start compare for test data set")
    start = time.time()
    result_test = comp.compute(pairs_test, dataDBLP_test, dataScholar_test)
    print("Compare on training data took %.2fs" % (time.time() - start))

    variants = 4
    matches = []
    for i in range(variants):
        classification_variant = i
        if classification_variant == 0:
            match = result_test[result_test[0]+result_test[1]+result_test[2]>2].index
            print("simple classifier")
            # simple classifier: add the values and use a threshold of 2
            matches.append((config_description, "Basic", match))
        elif classification_variant == 1:
            classifier = recordlinkage.LogisticRegressionClassifier()
            classifier.fit(result_train, links_train)
            match = classifier.predict(result_test)
            print("logistic regression classifier")
            matches.append((config_description, "Log", match))
        elif classification_variant == 2:
            classifier = recordlinkage.SVMClassifier()
            classifier.fit(result_train, links_train)
            match = classifier.predict(result_test)
            print("svm classifier")
            matches.append((config_description, "SVM", match))
        elif classification_variant == 3:
            classifier = recordlinkage.KMeansClassifier()
            match = classifier.fit_predict(result_test)
            print("KMeans classifier")
            matches.append((config_description, "KMeans", match))

        if debug:
            _, _, lastMatch = matches[-1]
            print("%d matches" % len(lastMatch))
            print_experiment_evaluation(lastMatch)
            # display(matches)

    return matches

indexer_variants = 2
comparison_variants = 2

matches = []
for indexer in range(indexer_variants):
    for comparer in range(comparison_variants):
        matches += run_experiment(indexer, comparer, 0)

print(matches)

# %% [markdown]
# ## 6. Evaluation
# 
# We use again the recordlinkage package for calculating evaluation values of the results.

# %%

x_axis_labels = []
precisions = []
recalls = []
fscores = []

for (config, classifier, match) in matches:
    precision, recall, fscore = print_experiment_evaluation(match)
    
    x_axis_labels.append(config+classifier)
    precisions.append(precision)
    recalls.append(recall)
    fscores.append(fscore)

x_axis = range(len(matches))

fig, axs = plt.subplots(1, 2)
fig.set_figwidth(10)
fig.set_figheight(5)

# plot results of sortedneighbors index (3 neighbors)
axs[0].set_ylim(0,1)
axs[0].scatter(x_axis_labels[0:3], precisions[0:3], label="Precision")
axs[0].scatter(x_axis_labels[0:3], recalls[0:3], label="Recall")
axs[0].scatter(x_axis_labels[0:3], fscores[0:3], label="F-Score")
# axs[0].legend()
axs[0].tick_params(labelrotation=45)

# plot results of sortedneighbors index (3 neighbors)
axs[1].set_ylim(0,1)
axs[1].scatter(x_axis_labels[6:9], precisions[6:9], label="Precision")
axs[1].scatter(x_axis_labels[6:9], recalls[6:9], label="Recall")
axs[1].scatter(x_axis_labels[6:9], fscores[6:9], label="F-Score")
axs[1].legend(loc='lower right')
axs[1].tick_params(labelrotation=45)

plt.show()
plt.savefig('results.png')

# %%
