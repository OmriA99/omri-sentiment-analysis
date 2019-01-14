import sys
import numpy as np
import random

DEBUG = False

# How to generate batch?
# Do we want to naively iterate through all examples
# Or maybe sample uniformly from each class
def batch_generator(data, batch_size):
    """
    Generates the next batch
    """
    X, y = data

    if X.shape[0] != y.shape[0]:
        raise Exception("non matching dimensions for X ({}) and y ({})".format(
            X.shape[0], y.shape[0]))

    size = X.shape[0]

    i = 0
    while True:
        if i + batch_size <= size:
            yield X[i:i + batch_size], y[i:i + batch_size]
            i += batch_size
        else:
            if i < size:
                to_yield = X[i:size], y[i:size]
                num_left_to_yield = batch_size - (size - i)

                i = 0
                yield (np.concatenate((to_yield[0], X[i:i + num_left_to_yield]), axis=0),
                    np.concatenate((to_yield[1], y[i:i + num_left_to_yield]), axis=0))
                i += num_left_to_yield

            else:
                i = 0


def batch_generator_uniform_prob(data, batch_size, num_classes):
    """
    Generates the next batch
    """
    X, y, cls_ranges = sort_data(data, num_classes)

    if DEBUG:
        print("cls_ranges = {}".format(cls_ranges))

    if X.shape[0] != y.shape[0]:
        raise Exception("non matching dimensions for X ({}) and y ({})".format(
            X.shape[0], y.shape[0]))

    size = X.shape[0]

    i = 0
    last = 0
    while True:
        # X.shape[1] is max_seq_length
        Xs = np.zeros((batch_size, X.shape[1]))
        ys = np.zeros((batch_size, num_classes))
        if DEBUG:
            print(X.shape)
            print(Xs.shape)

        for i in range(last, last+batch_size): # change batch_size to size
            idx = i % batch_size
            label = i % num_classes
            start, end = cls_ranges[label]
            rand_idx = random.randint(start, end)
            Xs[idx,:] = X[rand_idx,:]
            ys[idx][label] = 1

            if DEBUG:
                print("[batch_generator_uniform_prob()], i = {}, range = {}, randint = {}".format(
                    idx, cls_ranges[label], rand_idx))
                # print(Xs[i], ys[i])

        last = i+1
        yield Xs, ys


def sort_data(data, num_classes):
    """
    Sorts the given data by labels

    :return: (X, y) sorted by labels (0,1,2,3...) and list containing
             index ranges for each class
    """
    X, y = data
    if DEBUG:
        print("[sort_data()], y.shape = {}".format(y.shape))

    sorted_indices = np.argsort(y)
    X = X[sorted_indices]
    y = y[sorted_indices]

    cls_ranges = []
    start = 0
    for cls in range(num_classes):
        end = np.argmax(y == (cls + 1))
        if cls == num_classes -1:
            end = len(y)
        cls_ranges.append((start, end-1))
        start = end

    return X, y, cls_ranges


def load_word_vectors(path):
    word_vectors = np.load(path)
    return word_vectors


def find_first_occ(arr, target_val):
    """
    Returns first occurrence of target_val in arr

    :param arr: array to search target_val in
    :param target_val: value to search its first occurrence in arr
    :return: index of 1st occurrence of target_val in arr (if no such occurrence return -1)
    """
    for idx, val in enumerate(arr):
        if val == target_val:
            return idx

    return -1


def get_lengths(X, padd_value):
    """
    Returns the lengths of the unpadded vector for each vector in X

    :param X: array of arrays to calculate their lengths
    :param padd_value: padding value
    :return: numpy array of lengths of unpadded parts in arrays of X
    """
    return np.argmax(X == 0, axis=1)

    lengths = []
    for x in X:
        # length = find_first_occ(x, padd_value)
        length = np.argmax(x == 0)
        if length == 0 and x[0] != 0:
        # if length == -1:
            length = max(x.shape)
        lengths.append(length)

    return np.array(lengths)
