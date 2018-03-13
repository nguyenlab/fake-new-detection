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

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    Generate a matrix
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)

    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    # change from (20, 7) --> (7, 20)
    return np.transpose(encoding)

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

class MemNet(object):
    def __init__(self,
                 vocab_size,
                 statement_size,
                 embedding_size,
                 char_size,
                 num_hops,
                 num_classes):

        # place holders
        tf.set_random_seed(config.seed)
        self.checkpoint_dir = ""
        self.input_x = tf.placeholder(tf.int32, shape=[None, statement_size, char_size], name="input_x")
        self.party = tf.placeholder(tf.int32, shape=[None, char_size], name="party")
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # initilize embedding
        with tf.name_scope("embedding"):
            self.word_embeddings = []
            for hop_i in range(num_hops + 1):
                self.word_embeddings.append(
                    tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size], mean=0, stddev=0.1),
                                dtype=tf.float32, name="embedding_{}".format(hop_i)))

            self.party_embedded = tf.nn.embedding_lookup(self.word_embeddings[0], self.party)

        self._encoding = tf.constant(position_encoding(char_size, embedding_size), name="encoding")
        # build the network
        u_0 = tf.reduce_sum(self.party_embedded*self._encoding, axis= 1)
        u = [u_0]
        # print(u_0)
        # print("u_0")
        for hop_i in range(num_hops):
            if hop_i == 0:
                u_k = u[0]  # shape (B, 5)
            else:
                u_k = u[-1]

            #input memory
            A = tf.nn.embedding_lookup(self.word_embeddings[hop_i], self.input_x)
            m_A = tf.reduce_sum(A*self._encoding, 2)
            #position_encoding(sentence_size, embedding_size)

            u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])

            dotted = tf.reduce_sum(m_A * u_temp, 2)  # we have matrix with different size

            probs = tf.nn.softmax(dotted)  # (?, 10)
            probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])

            # output memory
            C = tf.nn.embedding_lookup(self.word_embeddings[hop_i + 1], self.input_x)
            m_C = tf.reduce_sum(C*self._encoding, 2)
            c_temp = tf.transpose(m_C, [0, 2, 1])
            o_k = tf.reduce_sum(c_temp * probs_temp, 2)
            u_k1 = u_k + o_k
            u.append(u_k1)
        #self.u_drop_out = tf.nn.dropout(u[-1], keep_prob=self.dropout_keep_prob)
        self.scores = feed_forward(u[-1], embedding_size, num_classes, reuse=False)
        
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
            x_batch, party_batch, y_batch = zip(*batch)
            feed_dict = {
                self.input_x: x_batch,
                self.input_y: y_batch,
                self.party: party_batch,
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
            x_batch, party_batch, y_batch = zip(*batch)
            feed_dict = {
                self.input_x: x_batch,
                self.input_y: y_batch,
                self.party: party_batch,
                self.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [self.global_step, dev_summary_op, self.loss, self.accuracy],
                feed_dict)

            print("step %8d, loss %6.3f, acc %6.3f" % (step, loss, accuracy))
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

            # x_test, sq_len_test, ch_test, y_test = zip(*test_data)
            # dev_step(test_data, writer=dev_summary_writer)

def eval(test_data, test_y, batch_size, checkpoint_dir, binary=True):
    #x_test, party_test, y_test = zip(*test_data)
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

            party = graph.get_operation_by_name("party").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_utils.batch_iter(test_data, batch_size, shuffle=False)
            # Collect the predictions here
            all_predictions = []

            for batch in batches:
                x_batch, party_batch = zip(*batch)
                batch_predictions = sess.run(predictions, {input_x: x_batch,
                                                           party: party_batch,
                                                           dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if test_y is not None:
        if binary:
            avr = 'binary'
        else:
            avr = 'macro'
        y_test = np.argmax(test_y, axis=1)

        acc = accuracy_score(y_test, all_predictions)
        precision = precision_score(y_test, all_predictions, average=avr)
        recall = recall_score(y_test, all_predictions, average=avr)
        f1 = f1_score(y_test, all_predictions, average=avr)
        confusion = confusion_matrix(y_test, all_predictions)
        print("acc = {:05.3f}, precision= {:05.3f}, recall= {:05.3f}, f1= {:05.3f}".format(acc, precision, recall, f1))
        print("confusion")
        print(confusion)

    return acc, precision, recall, f1