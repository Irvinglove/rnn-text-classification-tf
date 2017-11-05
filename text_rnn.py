#coding=utf-8
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np


class TextRNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, rnn_size, batch_size, l2_reg_lambda=0.5, model='lstm', num_layers=3):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        with tf.name_scope("lstm"):
            # # 构建RNN基本单元RNNcell
            # if model == 'rnn':
            #     cell_fun = tf.contrib.rnn.BasicRNNCell
            # elif model == 'gru':
            #     cell_fun = tf.contrib.rnn.GRUCell
            # else:
            #     cell_fun = tf.contrib.rnn.BasicLSTMCell
            # cell = cell_fun(rnn_size, state_is_tuple=True)
            # 构建堆叠rnn，这里选用两层的rnn
            def get_a_cell():
                return tf.nn.rnn_cell.BasicRNNCell(rnn_size)

            # 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
            cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(num_layers)])
            initial_state = cell.zero_state(batch_size, tf.float32)

            outputs, last_state = tf.nn.dynamic_rnn(cell, self.embedded_chars, initial_state=initial_state)
            self.output = tf.reshape(outputs, [batch_size, -1])

        # Add dropout
        with tf.name_scope("dropout"):
            self.rnn_drop = tf.nn.dropout(self.output, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[sequence_length * rnn_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.rnn_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
