# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # DBLP Scholar: Record Linkage attacks
# 
# This Notebook implements a few sample record linkage attacks and calculates the accuracy, precision and recall.

import recordlinkage
import time
import os
import matplotlib.pyplot as plt
from recordlinkage import compare
# import utility functions for dealing with datasets
from utils import read_data, preproc_attributes, split_train_test
import numpy as np

# set debug flag:
debug = True

# %%

# read DBLP and Google Scholar dataset
dataDBLP, dataScholar, links = read_data(
    'DBLP1', 'Scholar', 'DBLP-Scholar_perfectMapping', debug)

if debug and "display" in dir():
    display(dataDBLP)
    display(dataScholar)
    display(links)

# change into figures folder for exports
os.chdir('figures')

# ## 2. Cleaning and Pre-Processing

# cleaning: bring all to lowercase, remove unwanted tokens
# preprocessing: add multiple phonetic encodings
dataDBLP = preproc_attributes(dataDBLP, ['title', 'authors', 'venue'])
dataScholar = preproc_attributes(dataScholar, ['title', 'authors', 'venue'])
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

def print_experiment_evaluation(matches, description):
    precision = 0
    recall = 0
    fscore = 0

    if len(matches) > 0:
        precision = recordlinkage.precision(links_test, matches)
        recall = recordlinkage.recall(links_test, matches)
        fscore = recordlinkage.fscore(links_test, matches) if recall+precision>0 else 0
    
    print(f"Configuration: {description}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-score: {fscore}")
    print(recordlinkage.confusion_matrix(links_test, matches))

    return precision, recall, fscore

def run_experiment(win_len, preproc, comparison_variant, run_only=None):
    # window length
    if win_len == 0:
        index_description = "block"
        indexer = recordlinkage.BlockIndex('year')
    elif win_len > 0:
        index_description = f"nb{win_len}"
        indexer = recordlinkage.SortedNeighbourhoodIndex('year', window=win_len)
    else:
        raise ValueError(f"Invalid window length {win_len}")
    pairs_train = indexer.index(dataDBLP_train, dataScholar_train)
    pairs_test = indexer.index(dataDBLP_test, dataScholar_test)
    if debug:
        print(f"Number of candidates (index={index_description}):")
        print(f"{len(pairs_train)} (train), {len(pairs_test)} (test)")

    # preprocessing
    if preproc == 0:
        print("No preprocesing")
        field_suffix = ""
        preproc_description = "none"
    elif preproc == 1:
        print("Cleaned fields")
        field_suffix = "_clean"
        preproc_description = "clean"
    elif preproc == 2:
        print("Soundex encoding")
        field_suffix = "_soundex"
        preproc_description = "soundex"
    elif preproc == 3:
        print("Nysiis encoding")
        field_suffix = "_nysiis"
        preproc_description = "nysiis"
    elif preproc == 4:
        print("Metaphone encoding")
        field_suffix = "_metaphone"
        preproc_description = "metaphone"
    elif preproc == 5:
        print("Match-rating encoding")
        field_suffix = "_match_rating"
        preproc_description = "match_rating"
    else:
        raise ValueError(f"Unknown preprocessing variant {preproc}")
    print(f"Preprocessing used: {preproc_description}")

    # comparator
    comp = recordlinkage.Compare()
    if comparison_variant == 0:
        comp_description = "exact"
        comp.add(compare.Exact('title'+field_suffix, 'title'+field_suffix))
        comp.add(compare.Exact('authors'+field_suffix, 'authors'+field_suffix))
        comp.add(compare.Exact('venue'+field_suffix, 'venue'+field_suffix))
    elif comparison_variant == 1:
        comp_description = "levenshtein"
        comp.add(compare.String('title'+field_suffix, 'title'+field_suffix, method='levenshtein'))
        comp.add(compare.String('authors'+field_suffix, 'authors'+field_suffix, method='levenshtein'))
        comp.add(compare.String('venue'+field_suffix, 'venue'+field_suffix, method='levenshtein'))
    elif comparison_variant == 2:
        comp_description = "damerau_levenshtein"
        comp.add(compare.String('title'+field_suffix, 'title'+field_suffix, method='damerau_levenshtein'))
        comp.add(compare.String('authors'+field_suffix, 'authors'+field_suffix, method='damerau_levenshtein'))
        comp.add(compare.String('venue'+field_suffix, 'venue'+field_suffix, method='damerau_levenshtein'))
    elif comparison_variant == 3:
        comp_description = "jaro"
        comp.add(compare.String('title'+field_suffix, 'title'+field_suffix, method='jaro'))
        comp.add(compare.String('authors'+field_suffix, 'authors'+field_suffix, method='jaro'))
        comp.add(compare.String('venue'+field_suffix, 'venue'+field_suffix, method='jaro'))
    elif comparison_variant == 4:
        comp_description = "jarowinkler"
        comp.add(compare.String('title'+field_suffix, 'title'+field_suffix, method='jarowinkler'))
        comp.add(compare.String('authors'+field_suffix, 'authors'+field_suffix, method='jarowinkler'))
        comp.add(compare.String('venue'+field_suffix, 'venue'+field_suffix, method='jarowinkler'))
    elif comparison_variant == 5:
        comp_description = "qgram"
        comp.add(compare.String('title'+field_suffix, 'title'+field_suffix, method='qgram'))
        comp.add(compare.String('authors'+field_suffix, 'authors'+field_suffix, method='qgram'))
        comp.add(compare.String('venue'+field_suffix, 'venue'+field_suffix, method='qgram'))
    elif comparison_variant == 6:
        comp_description = "cosine"
        comp.add(compare.String('title'+field_suffix, 'title'+field_suffix, method='cosine'))
        comp.add(compare.String('authors'+field_suffix, 'authors'+field_suffix, method='cosine'))
        comp.add(compare.String('venue'+field_suffix, 'venue'+field_suffix, method='cosine'))
    elif comparison_variant == 7:
        comp_description = "smith_waterman"
        comp.add(compare.String('title'+field_suffix, 'title'+field_suffix, method='smith_waterman'))
        comp.add(compare.String('authors'+field_suffix, 'authors'+field_suffix, method='smith_waterman'))
        comp.add(compare.String('venue'+field_suffix, 'venue'+field_suffix, method='smith_waterman'))
    else:
        raise ValueError(f"Unknown comparison variant {comparison_variant}")
    print(f"String comparison: {comp_description}")

    print("Start compare for training data set")
    start = time.time()
    result_train = comp.compute(pairs_train, dataDBLP_train, dataScholar_train)
    print("Compare on training data took %.2fs" % (time.time() - start))
    print("Start compare for test data set")
    start = time.time()
    result_test = comp.compute(pairs_test, dataDBLP_test, dataScholar_test)
    # save time compare for evaluation
    time_compare = time.time() - start
    print("Compare on test data took %.2fs" % (time_compare))

    matches = []
    for classifier_description in ['logreg', 'bayes', 'svm', 'kmeans', 'ecm']:
        # skip others if only one classifier is requested
        if run_only is not None and run_only != classifier_description:
            continue
        if classifier_description == 'logreg':
            print("Logistic Regression classifier")
            classifier = recordlinkage.LogisticRegressionClassifier()
            supervised = True
        elif classifier_description == 'bayes':
            print("Naive Bayes classifier")
            classifier = recordlinkage.NaiveBayesClassifier(binarize=0.75)
            supervised = True
        elif classifier_description == 'svm':
            print("Support Vector Machine classifier")
            classifier = recordlinkage.SVMClassifier()
            supervised = True
        elif classifier_description == 'kmeans':
            print("KMeans classifier")
            classifier = recordlinkage.KMeansClassifier()
            supervised = False
        elif classifier_description == 'ecm':
            print("ECM classifier")
            classifier = recordlinkage.ECMClassifier(binarize=0.75)
            supervised = False
        else:
            raise ValueError(f"Unknown classifier variant {classifier_description}")

        if supervised:
            start = time.time()
            classifier.fit(result_train, links_train)
            time_train = time.time() - start
            start = time.time()
            match = classifier.predict(result_test)
            time_classify = time.time() - start
        else:
            start = time.time()
            match = classifier.fit_predict(result_test)
            time_classify = time.time() - start
            time_train = 0
        matches.append((index_description, preproc_description, comp_description, classifier_description, match, 1000*time_compare, 1000*time_train, 1000*time_classify))

        if debug:
            print("%d matches" % len(match))
            print_experiment_evaluation(match, "-".join((index_description, preproc_description, comp_description)))

    return matches

# %%

# Helper function: plot three lists into a scatter plot
def plot_experiment(x_list, classes, vals, filename, label=None, logscale=False):
    # find maximum values
    max_v = max(vals)
    # find unique class values
    if classes is None:
        classes = [0] * len(x_list)
    unique_classes = list(set(classes))
    
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(5)

    if max_v <= 1 and not logscale:
        ax.set_ylim(0, 1)

    for cl in unique_classes:
        x_labels = []
        y_vals = []
        for i in range(len(classes)):
            if classes[i] == cl:
                x_labels.append(x_list[i])
                y_vals.append(vals[i])
        ax.scatter(x_labels, y_vals, label=cl)
    ax.tick_params(labelrotation=45)
    if len(unique_classes) > 1:
        ax.legend(loc='lower right')
    if logscale:
        ax.set_yscale('log')
    if label:
        ax.set_ylabel(label)
    plt.show()
    saveto = 'debug_'+filename if debug else filename
    plt.savefig(saveto, bbox_inches='tight')

# Helper function: plot three lists into a bar plot
def plot_experiment_bars(x_list, classes, vals, filename, label=None, logscale=False):
    # find maximum values
    max_v = max(vals)

    # find unique class values
    if classes is None:
        classes = [0] * len(x_list)
    unique_classes = list(set(classes))
    c_idx = lambda cval: unique_classes.index(cval)
    
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(5)

    # calculate width of bars
    ccount = len(unique_classes)
    width = .7/ccount
    print(width)

    # get x indices
    unique_x = list(set(x_list))
    x = np.arange(len(unique_x))
    x_idx = lambda xval: unique_x.index(xval)

    # calculate x-pos for xval and class
    if ccount == 1:
        delta0 = 0
    elif ccount == 2:
        delta0 = 0.175
    elif ccount == 3:
        delta0 = 0.25
    x_pos = lambda xval, cval: x_idx(xval) + width*c_idx(cval) - delta0

    if max_v <= 1 and not logscale:
        ax.set_ylim(0, 1)

    for cl in unique_classes:
        xs = []
        ys = []
        for i in range(len(x_list)):
            if cl != classes[i]: continue
            _x = x_pos(x_list[i], classes[i])
            _y = vals[i]
            xs.append(_x)
            ys.append(_y)
        ax.bar(xs, ys, width, label=cl)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_x)
    ax.tick_params(labelrotation=45)
    if len(unique_classes) > 1:
        ax.legend()
    if logscale:
        ax.set_yscale('log')
    if label:
        ax.set_ylabel(label)
    plt.show()
    saveto = 'debug_'+filename if debug else filename
    plt.savefig(saveto, bbox_inches='tight')

# %%

# Experiment 1: compare influence of preprocessing (using two comparison mechanisms)
if debug or not os.path.exists('eval_preprocessing.pdf'):
    matches = []
    for prepoc in range(6):
        for comp in [0, 3]:
            matches += run_experiment(0, prepoc, comp, 'svm')

    l_preproc = []
    l_comp = []
    l_fscore = []
    for (_, preproc_desc, comp_desc, _, match, _, _, _) in matches:
        _, _, fscore = print_experiment_evaluation(
            match, preproc_desc + comp_desc)
        l_preproc.append(preproc_desc)
        l_comp.append(comp_desc)
        l_fscore.append(fscore)
    plot_experiment_bars(l_preproc, l_comp, l_fscore, 'eval_preprocessing.pdf', 'F1-score')

# %%

# Experiment 2: compare influence of index (using one comparison mechanism)
if debug or not os.path.exists('eval_indexing.pdf') or not os.path.exists('eval_indexing2.pdf'):
    matches = []
    for win_len in [0, 1, 3, 5, 7, 9]:
        # using exact match with nysiis
        matches += run_experiment(win_len, 3, 0, 'svm')

    # plot metrics
    l_winlen = []
    l_metric = []
    l_score = []
    for (winlen, _, _, _, match, _, _, _) in matches:
        precision, recall, fscore = print_experiment_evaluation(
            match, winlen)
        l_winlen.append(winlen)
        l_metric.append('precision')
        l_score.append(precision)
        l_winlen.append(winlen)
        l_metric.append('recall')
        l_score.append(recall)
        l_winlen.append(winlen)
        l_metric.append('f1score')
        l_score.append(fscore)
    plot_experiment_bars(l_winlen, l_metric, l_score, 'eval_indexing.pdf')

    # plot time
    l_winlen = []
    l_step = []
    l_time = []
    for (winlen, _, _, _, match, time_compare, time_train, time_classify) in matches:
        precision, recall, fscore = print_experiment_evaluation(
            match, winlen)
        l_winlen.append(winlen)
        l_step.append('compare')
        l_time.append(time_compare)
        l_winlen.append(winlen)
        l_step.append('train')
        l_time.append(time_train)
        l_winlen.append(winlen)
        l_step.append('classify')
        l_time.append(time_classify)
    plot_experiment_bars(l_winlen, l_step, l_time, 'eval_indexing2.pdf', 'Runtime (ms)')
    
# %%

# Experiment 3: compare influence of comparison (using one classifier)
# also looking at the runtime
# only with debug set
if debug and (not os.path.exists('debug_eval_comparison.pdf') or not os.path.exists('debug_eval_comparison2.pdf')):
    matches = []
    for comp in range(8):
        matches += run_experiment(0, 1, comp, 'logreg')

    l_comp = []
    l_fscore = []
    l_time = []
    for (_, _, comp_description, _, match, time_compare, _, _) in matches:
        _, _, fscore = print_experiment_evaluation(match, comp_description)
        l_comp.append(comp_description)
        l_fscore.append(fscore)
        l_time.append(time_compare)
    plot_experiment_bars(l_comp, None, l_fscore, 'eval_comparison.pdf', 'F1-score')
    plot_experiment_bars(l_comp, None, l_time, 'eval_comparison2.pdf', 'Runtime (ms)', True)

# %%

# Experiment 4: compare influence of classifier (using two comparators)
# also looking at the runtime

if debug or not os.path.exists('eval_classifier.pdf') or not os.path.exists('eval_classifier2.pdf'):
    # run all classifiers with exact match and nysiis preprocessing
    matches = run_experiment(0, 3, 0)
    # run all classifiers with levenshtein and clean preprocessing
    matches += run_experiment(0, 1, 1)

    # plot fscore
    l_clf = []
    l_preproc = []
    l_fscore = []
    for (_, preproc_desc, comp_desc, clf_desc, match, _, _, _) in matches:
        _, _, fscore = print_experiment_evaluation(
            match, preproc_desc + comp_desc + clf_desc)
        l_clf.append(clf_desc)
        l_preproc.append(comp_desc+"_"+preproc_desc)
        l_fscore.append(fscore)
    plot_experiment_bars(l_clf, l_preproc, l_fscore, 'eval_classifier.pdf', 'F1-score')

    # plot runtime
    l_clf = []
    l_step = []
    l_time = []
    for (_, preproc_desc, comp_desc, clf_desc, match, _, time_train, time_classify) in matches:
        if comp_desc != 'exact':
            continue
        l_clf.append(clf_desc)
        l_step.append('train')
        l_time.append(time_train)
        l_clf.append(clf_desc)
        l_step.append('classify')
        l_time.append(time_classify)
    plot_experiment_bars(l_clf, l_step, l_time, 'eval_classifier2.pdf', 'Runtime (ms)', True)
