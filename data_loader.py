import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import re
import sys
import os

# Constants
BASE_DIR = "data"
TRAIN_DATA_PATH = "data/train.tsv"
TEST_DATA_PATH = "data/test.tsv"
NUM_CLASSES = 5
TEST_SET_FRACT = 0.2
STRIP_SPECIAL_CHARS = re.compile("[^A-Za-z0-9 ]+")
UNKOWN_WORD = 399999
WORD_TO_NUM_FILE = "embeddings/word_to_num.npy"
DEBUG = False
TEST_LOSS_CONVERGENCE = True
TEST_CONVERGENCE_NUM_EXAMPLES = 10000

# ~~ Helpers ~~
def clean_sentence(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(STRIP_SPECIAL_CHARS, "", string.lower())


def get_data_params(base_dir_path):
    """
    Returns data params of the given dataset

    :param base_dir_path: path to directory containing both train and test sets
    :return: data params dictionary
    """

    params = {}

    train_path = os.path.join(base_dir_path, "train.tsv")
    test_path = os.path.join(base_dir_path, "test.tsv")

    # path
    params["train_path"] = train_path
    params["test_path"] = test_path

    # max_seq_length
    train = pd.read_csv(train_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')

    # params["max_seq_length"] = max(train.Phrase.str.len().max(), test.Phrase.str.len().max())
    params["max_seq_length"] = 80


    # word_to_num_map
    words_arr = np.load(WORD_TO_NUM_FILE)
    word_to_num_map = {word.decode("UTF-8"): idx for (idx, word) in enumerate(words_arr)}
    params["word_to_num_map"] = word_to_num_map

    return params


def integerize_sentence(sentence, word_to_num_map, max_seq_len):
    """
    Pads, cleans and integrizes the given sentence

    :param sentence: sentence to be processed
    :param word_to_num_map: map from word to its corresponding int
    :param max_seq_len: length to pad to
    :return: processed sentence
    """
    integerized = np.zeros((max_seq_len), dtype='int32')
    sentence = clean_sentence(sentence)
    splitted = sentence.split()

    for (idx, word) in enumerate(splitted):
        if idx > max_seq_len:
            break
        try:
            integerized[idx] = word_to_num_map[word]
        except KeyError:
            integerized[idx] = UNKOWN_WORD

    return integerized


def process_inputs(X, data_params):
    """
    Processes inputs (sentence --> array of ints).
    Each word is converted into an int according to data_params["word_to_num_map"]

    :param X: data to be processed, pandas.Series of phrases
    :data_params: data params
    :return: processed data as 2d numpy array (phrase --> array of ints)
    """

    processed_X = np.zeros((len(X), data_params["max_seq_length"]), dtype=np.int32)

    for idx, sentence in enumerate(X):
        integerized = integerize_sentence(sentence,
            data_params["word_to_num_map"] ,data_params["max_seq_length"])

        if DEBUG:
            if idx == 33:
                print("idx = {} sentence = {}, integerized = {}".format(idx, sentence, integerized))

        processed_X[idx] = integerized

    return processed_X


def show_stats():
    """
    Print interesting data statistics
    """
    train = pd.read_csv(TRAIN_DATA_PATH, sep='\t')
    print("Train head")
    print(train.head())
    print("\n\n")

    test = pd.read_csv(TEST_DATA_PATH, sep='\t')
    print("Test head")
    print(test.head())
    print("\n\n")

    class_count = train['Sentiment'].value_counts()
    print("Class count")
    print(class_count)
    print("\n\n")

    x = np.array(class_count.index)
    y = np.array(class_count.values)
    plt.figure()
    sns.barplot(x,y)
    plt.xlabel('Sentiment ')
    plt.ylabel('Number of reviews ')
    plt.show()


    # Does the 'Average words per sentence' calculation correct?
    print('Number of sentences in training set: ',len(train['SentenceId'].unique()))
    print('Number of sentences in test set:',len(test['SentenceId'].unique()))
    print('Average words per sentence in train:',
        train.groupby('SentenceId')['Phrase'].count().mean())
    print('Average words per sentence in test:',
        test.groupby('SentenceId')['Phrase'].count().mean())


def load_data(data_params, one_hot_labels=True):
    """
    Loads data from data file + Split into train & test

    :param data_params: params of the data
    :return: trainset, testset
    """
    train = pd.read_csv(data_params["train_path"], sep='\t')

    # max_seq_len = data_params["max_seq_length"]

    X_values = train['Phrase']
    labels_values = train.Sentiment.values

    if TEST_LOSS_CONVERGENCE:
        X_values = X_values[0:TEST_CONVERGENCE_NUM_EXAMPLES]
        labels_values = labels_values[0:TEST_CONVERGENCE_NUM_EXAMPLES]

    if DEBUG:
        print("X_Values.shape = {}".format(X_values.shape))
        print("labels_values = {}".format(labels_values.shape))
        idx = 123
        print("idx = {}: {} --> {}".format(idx, X_values[idx], labels_values[idx]))

    X_values = process_inputs(X_values, data_params)

    # Convert into one hot vectors
    if one_hot_labels:
        labels = np.zeros((len(labels_values), NUM_CLASSES))
        labels[np.arange(len(labels_values)), labels_values] = 1
    else:
        labels = labels_values

    if DEBUG:
        idx = 125584
        print("idx = {}: X_Values[33] = {} --> labels[33] = {}".format(
        idx, X_values[idx], labels[idx]))

    X_train , X_eval , y_train , y_eval = train_test_split(X_values, labels, test_size = TEST_SET_FRACT)

    if DEBUG:
        print("X_train.shape = {}, y_train.shape = {}".format(
            X_train.shape, y_train.shape))
        print("X_test.shape = {}, y_test.shape = {}".format(
            X_eval.shape, y_eval.shape))

        idx = 5
        print("Eval: idx = {} | {} --> {}".format(idx, X_eval[idx], y_eval[idx]))

    return X_train, X_eval, y_train, y_eval


def main():
    data_params = get_data_params(BASE_DIR)
    # show_stats()
    if DEBUG:
        # word to idx
        word = 'brush'
        print("word_to_num_map['{}'] = {}".format(
            word, data_params['word_to_num_map'][word]))

        # idxs to words
        words_arr = np.load(WORD_TO_NUM_FILE)
        nums = [29, 19612, 1069, 641, 740, 15860] #3
        print("Num to Word:")
        for num in nums:
            print("{} --> {}".format(num, words_arr[num]))



    load_data(data_params)


if __name__ == '__main__':
    main()
