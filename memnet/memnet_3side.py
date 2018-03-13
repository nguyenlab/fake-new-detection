import tensorflow as tf
import data_utils
import datetime
import time
import os
import numpy as np
import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

num_checkpoints = 5
checkpoint_every = 100


def add_mask(h, sequence_lenth, max_sequence, pad_value=-np.inf):
    ones = tf.ones_like(h, dtype=tf.float32)
    pad_values = pad_value * ones

    mask2d = tf.sequence_mask(lengths=sequence_lenth, maxlen=max_sequence)  # act as condition
    h_mask = tf.where(mask2d, h, pad_values)

    return h_mask

def _attend_dot(h, u):
    """Get the attend score of vector h and u of different dimensions. 
    This computation requires h and u to have the same vector_size

    :param h: 3-d matrix of shape (batch_size, max_len, vector_size)
    :param u: 2-d matrix of shape (batch_size, vector_size)
    :return: matrix of shape (batch_size, max_len)
    """

    max_len = h.shape[1]

    u = tf.expand_dims(u, 2)
    score = tf.matmul(h, u)
    score = tf.reshape(score, shape=tf.stack([-1, max_len]))
    p = tf.nn.softmax(score, dim=1)

    return p


def attend_concate(h, u, sequence_length, H, U, A, reuse, name):
    """
    perform a general type of attention based on agregation
    :param h: of shape (B, T, H), to multiply with (H, A) 
    :param u: of shape (B, U) to multiply with (U, A)
    :return: v_t*tanh(h*W_h + u*W_u + b)
    """
    T = h.shape[1]
    with tf.variable_scope(name, reuse=reuse):  # change to tf.get_variable()
        initializer = tf.random_normal_initializer(0.0, 0.1)
        # W_h = tf.Variable(tf.truncated_normal(shape=[H, A], mean=0, stddev=0.1), name="W_h")
        # W_u = tf.Variable(tf.truncated_normal(shape=[U, A], mean=0, stddev=0.1), name="W_u")
        W_h = tf.get_variable(name="W_h",
                              shape=[H, A],
                              dtype=tf.float32,
                              initializer=initializer)

        W_u = tf.get_variable(name="W_u",
                              shape=[U, A],
                              dtype=tf.float32,
                              initializer=initializer)
        # b = tf.get_variable(name= "b")
        b = tf.Variable(tf.constant(0.1, shape=[A], dtype=tf.float32), name="b")

        agregate = tf.tanh(tf.tensordot(h, W_h, axes=1) + tf.expand_dims((tf.matmul(u, W_u) + b), 1))
        v = tf.Variable(tf.truncated_normal(shape=[A], mean=0, stddev=0.1))
        raw_score = tf.tensordot(agregate, v, axes=1)
        masked_attend = add_mask(raw_score, sequence_length, T)
        p = tf.nn.softmax(masked_attend)

    return p


def weight_vector(c, p):
    """ weight vector c by a probability vector p
    :param c: vector to be weighted
    :param p: attention score
    :return: 
    """
    vector_size = c.shape[2]
    shape = c.shape
    p = tf.expand_dims(p, 1)
    weighted_vector = tf.matmul(p, c)
    weighted_vector = tf.reshape(weighted_vector, shape=tf.stack([-1, vector_size]))

    return weighted_vector


def feed_forward(input, n_in, n_out, reuse, name="fc"):
    initializer = tf.random_normal_initializer(0.0, 0.1)

    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(
            "W",
            shape=[n_in, n_out],
            initializer=initializer)

        b = tf.Variable(tf.constant(0.1, shape=[n_out]), name="bias")

        pre_act = tf.nn.xw_plus_b(input, W, b)
        act = tf.nn.relu(pre_act)

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)

        return act


def agrregate(input1, input2, n1_in, n2_in, n_out, reuse, name="agrregate"):
    """ 
    :param input1: matrix of shape (B, n1) 
    :param input2: matrix of shape (B, n2)
    :param n1_in: 
    :param n2_in: 
    :param n_out: 
    :param reuse: 
    :param name: 
    :return: tf.nn.relu(input1 * W1 + input2 * W2 + b) 
    """
    # initializer = tf.random_normal_initializer(0.0, 0.1)

    with tf.variable_scope(name, reuse=reuse):
        W1 = tf.get_variable(
            "W1",
            shape=[n1_in, n_out],
            initializer=tf.random_normal_initializer(0.0, 0.1))

        W2 = tf.get_variable(
            "W2",
            shape=[n2_in, n_out],
            initializer=tf.random_normal_initializer(0.0, 0.1))

        b = tf.Variable(tf.constant(0.1, shape=[n_out]), name="bias")

        pre_act = tf.matmul(input1, W1) + tf.matmul(input2, W2) + b
        act = tf.nn.relu(pre_act)

        # tf.summary.histogram("W1", W1)
        # tf.summary.histogram("W2", W2)
        # tf.summary.histogram("bias", b)
        # tf.summary.histogram("activations", act)

        return act


class MemNet(object):
    def __init__(self,
                 vocab_size,
                 statment_size,
                 word_embedding_size,
                 side1_vocab_size,
                 side2_vocab_size,
                 side_size,
                 ch_size,
                 num_hops,
                 num_classes,
                 attention_size):

        # place holders
        self.checkpoint_dir = ""
        self.input_x = tf.placeholder(tf.int32, shape=[None, statment_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name="input_y")
        self.sequence_length = tf.placeholder(tf.float32, shape=[None], name="sequence_length")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # side information

        self.side_info1 = tf.placeholder(tf.int32, shape=[None], name="side_info1")
        self.side_info2 = tf.placeholder(tf.int32, shape=[None], name="side_info2")
        self.input_ch = tf.placeholder(tf.float32, shape= [None, ch_size], name="credit_history")
        # self.input_sj = tf.placeholder(tf.int32, shape=[None], name="input_subject")
        # self.input_sp = tf.placeholder(tf.int32, shape=[None], name="input_speaker")
        # self.input_st = tf.placeholder(tf.int32, shape=[None], name="input_state")
        # self.input_pa = tf.placeholder(tf.int32, shape=[None], name="input_party")
        # self.input_lc = tf.placeholder(tf.int32, shape=[None], name="input_location")
        #self.input_ch = tf.placeholder(tf.float32, shape=[None, ch_size], name="credit_history")
        tf.set_random_seed(config.seed)
        # initilize embedding
        with tf.name_scope("embedding"):
            self.word_embeddings = []
            for hop_i in range(num_hops + 1):
                self.word_embeddings.append(
                    tf.Variable(tf.truncated_normal(shape=[vocab_size, word_embedding_size], mean=0, stddev=0.1),
                                dtype=tf.float32, name="embedding_{}".format(hop_i)))

            self.side_embedding1 = tf.Variable(tf.truncated_normal(shape=[side1_vocab_size, side_size], stddev=0.1),
                                                dtype=tf.float32,
                                                name="tp_embedding")

            self.side_embedded1 = tf.nn.embedding_lookup(params=self.side_embedding1, ids=self.side_info1)

            self.side_embedding2 = tf.Variable(tf.truncated_normal(shape=[side2_vocab_size, side_size], stddev=0.1),
                                              dtype=tf.float32,
                                              name="tp_embedding")

            self.side_embedded2 = tf.nn.embedding_lookup(params=self.side_embedding2, ids=self.side_info2)
                # self.tp_embedding = tf.Variable(tf.truncated_normal(shape=[None], stddev=0.1), dtype=tf.float32,
                #                                 name="tp_embedding")
                # self.sj_embedding = tf.Variable(tf.truncated_normal(shape=[None], stddev=0.1), dtype=tf.float32,
                #                                 name="sj_embedding")
                # self.sp_embedding = tf.Variable(tf.truncated_normal(shape=[None], stddev=0.1), dtype=tf.float32,
                #                                 name="sp_embedding")
                # self.st_embedding = tf.Variable(tf.truncated_normal(shape=[None], stddev=0.1), dtype=tf.float32,
                #                                 name="st_embedding")
                # self.pa_embedding = tf.Variable(tf.truncated_normal(shape=[None], stddev=0.1), dtype=tf.float32,
                #                                 name="pa_embedding")
                # self.ch_embedding = tf.Variable(tf.truncated_normal(shape=[None], stddev=0.1), dtype=tf.float32,
                #                                 name="ch_embedding")
                # self.lc_embedding = tf.Variable(tf.truncated_normal(shape=[None], stddev=0.1), dtype=tf.float32,
                #                                 name="lc_embedding")

        # build the network
        # a version of adding weights

        u_concated = tf.concat([self.input_ch, self.side_embedded1, self.side_embedded2], axis= 1)
        concated_size = ch_size + side_size*2
        u = [u_concated]

        #u1.append(self.input_ch)
        #u2.append(self.side_embedded)

        for hop_i in range(num_hops):
            if hop_i == 0:
                u_k = u[0]  # shape (B, 5)
            else:
                u_k = u[-1]

            A = tf.nn.embedding_lookup(self.word_embeddings[hop_i], self.input_x)  # (?, statement_size, embedding_size)
            C = tf.nn.embedding_lookup(self.word_embeddings[hop_i + 1],
                                       self.input_x)  # (?, statement_size, embedding_size)

            if hop_i == 0:
                reuse = False
            else:
                reuse = True

            p = attend_concate(A, u_k, self.sequence_length, word_embedding_size, concated_size, attention_size, reuse=reuse,
                               name="attend".format(hop_i))

            o_k = weight_vector(C, p)

            u_k1 = agrregate(u_k, o_k, concated_size, word_embedding_size, concated_size, reuse=reuse,
                             name='agrregate'.format(hop_i))
            u.append(u_k1)

        self.u_drop_out = tf.nn.dropout(u[-1], keep_prob=self.dropout_keep_prob)
        self.scores = feed_forward(self.u_drop_out, concated_size, num_classes, reuse=False)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(losses)
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope("output"):
            y_pred = tf.argmax(self.scores, axis=1, name="predictions")
            correct_preds = tf.equal(y_pred, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"), name="accuracy")
            tf.summary.scalar("accuracy", self.accuracy)

    def train(self, sess, train_data, dev_data, test_data, starter_learning_rate, num_epochs, batch_size, dropout_keep):

        learning_rate = starter_learning_rate
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        with tf.name_scope("train"):
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        sess.run(tf.global_variables_initializer())

        # output dir
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # summary op
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        train_summary_op = tf.summary.merge_all()
        dev_summary_op = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter(train_summary_dir)
        train_summary_writer.add_graph(sess.graph)
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir)
        dev_summary_writer.add_graph(sess.graph)
        # Checkpoint
        # Output directory for models and summaries

        self.checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        # saver
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        def train_step(batch):
            x_batch, sq_len_batch, ch_batch, side1_batch, side2_batch, y_batch = zip(*batch)
            feed_dict = {
                self.input_x: x_batch,
                self.input_y: y_batch,
                self.sequence_length: sq_len_batch,
                self.side_info1: side1_batch,
                self.side_info2: side2_batch,
                self.input_ch: ch_batch,
                self.dropout_keep_prob: dropout_keep
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, self.global_step, train_summary_op, self.loss, self.accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {}, acc {}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

            return loss

        def dev_step(batch, writer=None):
            x_batch, sq_len_batch, ch_batch, side1_batch, side2_batch, y_batch = zip(*batch)
            feed_dict = {
                self.input_x: x_batch,
                self.sequence_length: sq_len_batch,
                self.input_y: y_batch,
                self.side_info1: side1_batch,
                self.side_info2: side2_batch,
                self.input_ch: ch_batch,
                self.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [self.global_step, dev_summary_op, self.loss, self.accuracy],
                feed_dict)

            print("step %8d, loss %4.3f, acc %4.3f" % (step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Training loop. For each epoch...
        for i in range(num_epochs):
            batches = data_utils.batch_iter(train_data, batch_size, num_epochs)

            for batch in batches:
                train_step(batch)
                current_step = tf.train.global_step(sess, self.global_step)

                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)

            print("\nEpoch {}. Evaluation on dev set:".format(i))

            dev_step(dev_data, writer=dev_summary_writer)
            dev_step(test_data)

def eval(test_data, batch_size, checkpoint_dir, binary=True):
    x_test, sq_length_test, ch_test, side1_test, side2_test, y_test = zip(*test_data)
    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    print("checkpoint_file: ", checkpoint_file)
    graph = tf.Graph()
    with graph.as_default():
        # session_conf = tf.ConfigProto(
        #     allow_soft_placement=True,
        #     log_device_placement=False)
        # sess = tf.Session(config=session_conf)
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            sequence_length = graph.get_operation_by_name("sequence_length").outputs[0]
            side1_info = graph.get_operation_by_name("side_info1").outputs[0]
            side2_info = graph.get_operation_by_name("side_info2").outputs[0]
            ch = graph.get_operation_by_name("credit_history").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_utils.batch_iter(list(zip(x_test, sq_length_test, ch_test, side1_test, side2_test)), batch_size, shuffle=False)
            # Collect the predictions here
            all_predictions = []

            for batch in batches:
                x_batch, sq_length_batch, ch_batch, side1_batch, side2_batch = zip(*batch)
                batch_predictions = sess.run(predictions, {input_x: x_batch,
                                                           sequence_length: sq_length_batch,
                                                           side1_info: side1_batch,
                                                           side2_info: side2_batch,
                                                           ch: ch_batch,
                                                           dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        if binary:
            avr = 'binary'
        else:
            avr = 'macro'
        y_test = np.argmax(y_test, axis=1)

        acc = accuracy_score(y_test, all_predictions)
        precision = precision_score(y_test, all_predictions, average=avr)
        recall = recall_score(y_test, all_predictions, average=avr)
        f1 = f1_score(y_test, all_predictions, average=avr)
        confusion = confusion_matrix(y_test, all_predictions)
        print("acc = {:04.3f}, precision= {:04.5f}, recall= {:04.5f}, f1= {:04.5f}".format(acc, precision, recall, f1))
        print("confusion")
        print(confusion)

    return acc, precision, recall, f1