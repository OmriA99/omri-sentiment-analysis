import sys
import os
import datetime
import tensorflow as tf
import numpy as np
import math
import tqdm
from utils import batch_generator, load_word_vectors, get_lengths, batch_generator_uniform_prob
import data_loader

# params
NUM_CLASSES = 5
LSTM_NUM_UNITS = 64
D_KEEP_PROB = 0.5
DATA_BASE_DIR = "data"
LOGS_BASE_DIR = "logs"
MODELS_BASE_DIR = "models"
WORD_VECTORS_PATH = "embeddings/word_vectors.npy"
PADD_VAL = 0
DEBUG = False

# tests
# test results without dynamic_rnn's copy-through state
DYN_RNN_COPY_THROUGH_STATE = True
# test Dropout
USE_DROPOUT = False
# test uniform distribution batch generation
USE_ONE_HOT_LABELS = True

def evaluate():
    """
    Given model & params - evaluate model's performance by
    running it on the evaluation set
    """

def train():
    """
    Build and Train model by given params
    """

    # params
    # assigned after loading data
    max_seq_length = None
    exp_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    keep_prob = 0.5
    n_hidden = 64
    num_classes = 5
    learning_rate = 1e-3
    model_save_path = os.path.join(MODELS_BASE_DIR, exp_name + '.cpkt')
    train_iterations = 100000
    eval_iterations = None
    batch_size = 24
    word_vector_dim = 300

    # ************** Pre-Model **************
    # Load data
    data_params = data_loader.get_data_params(DATA_BASE_DIR)
    max_seq_length = data_params["max_seq_length"]
    X_train, X_eval, y_train, y_eval = data_loader.load_data(data_params, one_hot_labels=USE_ONE_HOT_LABELS)
    print("==> Loaded data")

    eval_iterations = math.ceil(float(X_eval.shape[0]) / batch_size)

    # Load GloVe embbeding vectors
    word_vectors = load_word_vectors(WORD_VECTORS_PATH)

    # Batch generators
    #train_batch_generator = batch_generator_uniform_prob((X_train, y_train), batch_size, num_classes)
    #eval_batch_generator = batch_generator_uniform_prob((X_eval, y_eval), batch_size, num_classes)
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

    # lstm cell
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    if USE_DROPOUT:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prob)
    # Do we need the state tuple? Because we don't want the cell to be
    # initialized with the state from previous sentence
    ## rnn_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(init_state[0], init_state[1])

    if DYN_RNN_COPY_THROUGH_STATE:
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32, sequence_length=input_data_lengths)
    else:
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

    # output layer
    weight = tf.Variable(tf.truncated_normal([n_hidden, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

    # Let's try this logic
    outputs = tf.transpose(outputs, [1, 0, 2]) # max_seq_length, batch_size, word_vector_dim
    last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # Metrics
    # Should we reduce_mean?
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Summaries
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = os.path.join(LOGS_BASE_DIR, exp_name, "")

    # ************** Train **************
    print("Run 'tensorboard --logdir={}' to checkout tensorboard logs.".format(os.path.abspath(logdir)))
    print("==> training")


    best_accuracy = -1

    # Train
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(os.path.join(logdir, "train"))
        eval_writer = tf.summary.FileWriter(os.path.join(logdir, "evaluation"))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # Py2.7 or Py3 (if 2.7 --> Change to xrange)
        for iteration in tqdm.tqdm(range(train_iterations)):
            # shoudn't get exception, but check this
            # pass also
            X, y = next(train_batch_generator)
            X_lengths = get_lengths(X, PADD_VAL)
            if DEBUG:
                print("X.shape = {}, X_lengths.shape = {}".format(X.shape, X_lengths.shape))
                print("y.shape = {}".format(y.shape))
                print("type(X) = {}, type(X_lengths) = {}".format(X.dtype, X_lengths.dtype))
                idx = 3
                print("X[:{0}], X_length[:{0}]".format(idx))
                print(X[:idx])
                print(X_lengths[:idx])

            sess.run([optimizer], feed_dict={input_data: X, labels: y, input_data_lengths: X_lengths})

            # Write summary
            if (iteration % 30 == 0):
                _summary, = sess.run([merged], feed_dict={input_data: X, labels: y, input_data_lengths: X_lengths})
                train_writer.add_summary(_summary, iteration)

            # evaluate the network every 1,000 iterations
            if (iteration % 1000 == 0 and iteration != 0):
                total_accuracy = 0
                for eval_iteration in tqdm.tqdm(range(eval_iterations)):
                    X, y = next(eval_batch_generator)
                    X_lengths = get_lengths(X, PADD_VAL)
                    _accuracy, _summary = sess.run([accuracy, merged], feed_dict={input_data: X, labels: y,
                        input_data_lengths: X_lengths})
                    total_accuracy += _accuracy

                average_accuracy = total_accuracy / eval_iterations
                print("accuracy = {}".format(average_accuracy))
                if average_accuracy > best_accuracy:
                    print("Best model!")

                    save_path = saver.save(sess, model_save_path, global_step=iteration)
                    print("saved to %s" % save_path)

                    best_accuracy = average_accuracy

        eval_writer.close()
        train_writer.close()


def main():
    tf.reset_default_graph()
    train()


if __name__ == '__main__':
    main()
