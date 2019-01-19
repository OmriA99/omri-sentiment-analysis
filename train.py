import sys

import os

import datetime

import tensorflow as tf

import numpy as np

import math

import tqdm

from utils import batch_generator, load_word_vectors, get_lengths, batch_generator_uniform_prob

import data_loader

import argparse



# params

NUM_CLASSES = 5

LSTM_NUM_UNITS = 64

D_KEEP_PROB = 0.2

DATA_BASE_DIR = "data"

LOGS_BASE_DIR = "logs"

MODELS_BASE_DIR = "models"

WORD_VECTORS_PATH = "embeddings/word_vectors.npy"

PADD_VAL = 0

DEBUG = False




DYN_RNN_COPY_THROUGH_STATE = True

USE_DROPOUT = True

USE_UNIFORM_PROB = True

USE_ONE_HOT_LABELS = not USE_UNIFORM_PROB





def labelDistrubution(y):

    buckets = np.zeros([5])

    for arr in y:

        if USE_ONE_HOT_LABELS:

            for idx, val in enumerate(arr):

                if val == 1:

                    break

            buckets[idx] += 1

        else:

            buckets[eval] += 1

    return buckets



def evaluate(params):

    """

    Given model & params - evaluate model's performance by

    running it on the evaluation set

    """
    eval_batch_generator = params["eval_batch_generator"]
    accuracy = params["accuracy"]
    merged = params["merged"]
    input_data = params["input_data"]
    labels = params["labels"]
    input_data_lengths = params["input_data_lengths"]
    save_path = params["save_path"]
    model_save_path = params["model_save_path"]
    accuracy_file_path = params["accuracy_file_path"]
    
    total_accuracy = 0
    for eval_iteration in tqdm.tqdm(range(eval_iterations)):
        X, y = next(eval_batch_generator)
        X_lengths = get_lengths(X, PADD_VAL)
        _accuracy, _summary = sess.run([accuracy, merged], feed_dict={input_data: X, labels: y,
            input_data_lengths: X_lengths})
        total_accuracy += _accuracy

    average_accuracy = total_accuracy / eval_iterations
    print("({}) accuracy = {}".format(int(iteration/1000), average_accuracy))
    if average_accuracy > best_accuracy:
        print("Best model!")

        # save_path = saver.save(sess, model_save_path, global_step=iteration)
        save_path = saver.save(sess, model_save_path)
        print("saved to %s" % save_path)
        
        best_accuracy = average_accuracy
        with open(accuracy_file_path, 'a+') as f:
            f.write("{}".format(best_accuracy))



def train(args):

    """

    Build and Train model by given params

    """



    exp_name = args.exp_name
    evaluate_only = args.evaulate_only
    

    max_seq_length = None

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    keep_prob = 0.5

    n_hidden = 128

    num_layers = 2

    num_classes = 5

    learning_rate = 1e-3
    exp_base_dir = os.path.join(MODELS_BASE_DIR, exp_name)
    if not os.path.exists(exp_base_dir):
        os.makedirs(exp_base_dir)

    accuracy_file_path = os.path.join(exp_base_dir, "accuracies.txt")

    model_save_path = os.path.join(exp_base_dir, 'model.cpkt')

    train_iterations = 100000

    eval_iterations = None

    batch_size = 24

    word_vector_dim = 300
    
    if evaluate_only:
        if not os.path.exists(model_save_path):
            print("[-] Trying to evaluate a non-existing model!")
            return



    # ************** Pre-Model **************

    # Load data

    data_params = data_loader.get_data_params(DATA_BASE_DIR)

    max_seq_length = data_params["max_seq_length"]

    X_train, X_eval, y_train, y_eval = data_loader.load_data(data_params, one_hot_labels=USE_ONE_HOT_LABELS)



    print("==> Loaded data [X_train = {}, y_train = {}, X_eval = {}, y_eval = {}]".format(

        X_train.shape, y_train.shape, X_eval.shape, y_eval.shape

        ))



    eval_iterations = math.ceil(float(X_eval.shape[0]) / batch_size)



    # Load GloVe embbeding vectors

    word_vectors = load_word_vectors(WORD_VECTORS_PATH)



    # Batch generators

    if USE_UNIFORM_PROB:

        train_batch_generator = batch_generator_uniform_prob((X_train, y_train), batch_size, num_classes)

        eval_batch_generator = batch_generator_uniform_prob((X_eval, y_eval), batch_size, num_classes)

    else:

        train_batch_generator = batch_generator((X_train, y_train), batch_size)

        eval_batch_generator = batch_generator((X_eval, y_eval), batch_size)



    # ************** Model **************

    # placeholders

    labels = tf.placeholder(tf.float32, [None, num_classes])

    input_data = tf.placeholder(tf.int32, [None, max_seq_length])

    input_data_lengths = tf.placeholder(tf.int32, batch_size)



    # data processing

    data = tf.Variable(tf.zeros([batch_size, max_seq_length,

        word_vector_dim]), dtype=tf.float32)



    data = tf.nn.embedding_lookup(word_vectors, input_data)



    n_units = [n_hidden] * num_layers

    stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(

        [tf.nn.rnn_cell.DropoutWrapper(

            tf.nn.rnn_cell.LSTMCell(n), output_keep_prob=keep_prob) for n in n_units])



    if DYN_RNN_COPY_THROUGH_STATE:

        outputs, last_state = tf.nn.dynamic_rnn(stacked_rnn_cell, data, dtype=tf.float32, sequence_length=input_data_lengths)

    else:

        outputs, _ = tf.nn.dynamic_rnn(stacked_rnn_cell, data, dtype=tf.float32)



    # output layer

    weight = tf.Variable(tf.truncated_normal([n_units[num_layers-1], num_classes]))

    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))



    # we want the state of the last layer

    # state is a LSTMStateTuple that cotains state's c,h

    # state[1] is the 'h' part of it

    last = last_state[num_layers-1][1]

    prediction = (tf.matmul(last, weight) + bias)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



    # Metrics

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)



    # Summaries

    tf.summary.scalar('Loss', loss)

    tf.summary.scalar('Accuracy', accuracy)

    merged = tf.summary.merge_all()

    logdir = os.path.join(LOGS_BASE_DIR, exp_name, current_time, "")



    # ************** Train **************

    print("Run 'tensorboard --logdir={}' to checkout tensorboard logs.".format(os.path.abspath(logdir)))

    print("==> training")





    best_accuracy = -1



    # Train

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(os.path.join(logdir, "train"))

        # eval_writer = tf.summary.FileWriter(os.path.join(logdir, "evaluation"))



        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        
        if evaluate_only:
            saver.restore(sess, model_save_path)
            print("Model restored")
            params = {}
            params["eval_batch_generator"] = eval_batch_generator
            params["accuracy"] = accuracy
            params["merged"] = merged
            params["input_data"] = input_data
            params["labels"] = labels
            params["input_data_lengths"] = input_data_lengths
            params["best_accuracy"] = best_accuracy
            params["save_path"] = save_path
            params["model_save_path"] = model_save_path
            params["accuracy_file_path"] = accuracy_file_path
            evaluate(params)
        
        else:

            for iteration in tqdm.tqdm(range(train_iterations)):

                X, y = next(train_batch_generator)



                X_lengths = get_lengths(X, PADD_VAL)

                if DEBUG:

                    print("X.shape = {}, X_lengths.shape = {}".format(X.shape, X_lengths.shape))

                    print("y.shape = {}".format(y.shape))

                    print("type(X) = {}, type(X_lengths) = {}".format(X.dtype, X_lengths.dtype))

                    idx = 3



                sess.run([optimizer], feed_dict={input_data: X, labels: y, input_data_lengths: X_lengths})



                # Write summary

                if (iteration % 30 == 0):

                    _summary, = sess.run([merged], feed_dict={input_data: X, labels: y, input_data_lengths: X_lengths})

                    train_writer.add_summary(_summary, iteration)



                # evaluate the network every 1,000 iterations

                if (iteration % 1000 == 0 and iteration != 0):
                    params = {}
                    params["eval_batch_generator"] = eval_batch_generator
                    params["accuracy"] = accuracy
                    params["merged"] = merged
                    params["input_data"] = input_data
                    params["labels"] = labels
                    params["input_data_lengths"] = input_data_lengths
                    params["best_accuracy"] = best_accuracy
                    params["model_save_path"] = model_save_path
                    params["accuracy_file_path"] = accuracy_file_path
                    evaluate(params)



        # eval_writer.close()

        train_writer.close()





def main(args):

    tf.reset_default_graph()

    train(args)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(

    description="Model arguments"

    )

    parser.add_argument('-e', '--exp-name', required=True, type=str, help='Experiment name')
    parser.add_argument('-ev', '--evaulate-only', action='store_true', help='evaluate given model')

    main(parser.parse_args())

