import tensorflow as tf
import data_utils
import datetime
import time
import os
import numpy as np
import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matrix_utils import *

num_checkpoints = 5
checkpoint_every = 100

def print_list(list):
    for element in list:
        print(element)

def feed_forward(input, n_in, n_out, reuse, name="fc"):

    initializer = tf.random_normal_initializer(0.0, 0.1)

    with tf.variable_scope(name, reuse= reuse):
        W = tf.get_variable(
            "W",
            shape=[n_in, n_out],
            initializer= initializer)

        b = tf.Variable(tf.constant(0.1, shape=[n_out]), name="bias")

        pre_act = tf.nn.xw_plus_b(input, W, b)
        #act = tf.nn.relu(pre_act)

        #tf.summary.histogram("weights", W)
        #tf.summary.histogram("biases", b)
        #tf.summary.histogram("activations", act)

        return pre_act

class SingAtt(object):
    def __init__(self,
                 vocab_size,
                 statment_size,
                 word_embedding_size,
                 topic_size,
                 location_size,
                 side_size,
                 num_classes):

        # place holders
        self.checkpoint_dir = ""
        self.statement = tf.placeholder(tf.int32, shape=[None, statment_size], name="statement")
        self.topic = tf.placeholder(tf.int32, shape=[None, topic_size], name="topic")
        self.speaker = tf.placeholder(tf.int32, shape=[None], name="speaker")
        self.state = tf.placeholder(tf.int32, shape=[None], name="state")
        self.party = tf.placeholder(tf.int32, shape=[None], name="party")
        self.job = tf.placeholder(tf.int32, shape=[None], name="job")
        self.location = tf.placeholder(tf.int32, shape=[None, location_size], name="location")
        self.ch = tf.placeholder(tf.float32, shape=[None, side_size], name="credit_history")
        self.statement_sq = tf.placeholder(tf.float32, shape=[None], name="statement_sq")
        self.location_sq = tf.placeholder(tf.float32, shape=[None], name="location_sq")
        self.topic_sq = tf.placeholder(tf.float32, shape=[None], name="topic_sq")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name="input_y")

        tf.set_random_seed(config.seed)

        # embedding
        with tf.name_scope("embedding"):
            self.W_embedding = tf.Variable(tf.truncated_normal(shape=[vocab_size, word_embedding_size], mean=0, stddev=0.1),
                    dtype=tf.float32, name="embedding")

            self.statement_embedded = tf.nn.embedding_lookup(params= self.W_embedding, ids= self.statement)
            topic_embedded = tf.nn.embedding_lookup(params= self.W_embedding, ids= self.topic)
            speaker_embedded = tf.nn.embedding_lookup(params= self.W_embedding, ids= self.speaker)
            state_embedded = tf.nn.embedding_lookup(params=self.W_embedding, ids=self.state)
            party_embedded = tf.nn.embedding_lookup(params=self.W_embedding, ids=self.party)
            job_embedded = tf.nn.embedding_lookup(params=self.W_embedding, ids=self.job)
            location_embedded = tf.nn.embedding_lookup(params=self.W_embedding, ids=self.location)

        # sum topic and location
        topic_embedded = add_3dmask(topic_embedded, self.topic_sq, topic_size, word_embedding_size, mask_value=0)
        topic_embedded = tf.reduce_sum(topic_embedded, axis= 1)
        location_embedded = add_3dmask(location_embedded, self.location_sq, location_size, word_embedding_size, mask_value=0)
        location_embedded = tf.reduce_sum(location_embedded, axis=1)

        self.side_info = tf.stack([topic_embedded, speaker_embedded, job_embedded, state_embedded,
                      party_embedded, location_embedded, self.ch],
                     axis=1)

        self.side_sum = tf.reduce_sum(self.side_info, axis= 1)
        self.statement_sum = tf.reduce_sum(self.statement_embedded, axis= 1)

        self.statement_attend = base_attention(self.statement_embedded, self.side_sum, statment_size, word_embedding_size, self.statement_sq)
        self.side_attend = base_attention(self.side_info, self.statement_sum, 7,
                                          word_embedding_size, None)

        self.h = self.statement_attend

        self.scores = feed_forward(self.h, word_embedding_size, num_classes, reuse=False)

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(self.losses)
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
            statement, topic, speaker, state, party, job, location, ch, statement_sq, topic_sq, location_sq, y_batch \
                = zip(*batch)

            feed_dict = {
                self.statement: statement,
                self.topic: topic,
                self.speaker: speaker,
                self.state: state,
                self.party: party,
                self.job: job,
                self.location: location,
                self.ch: ch,
                self.statement_sq: statement_sq,
                self.topic_sq: topic_sq,
                self.location_sq: location_sq ,
                self.dropout_keep_prob: dropout_keep,
                self.input_y : y_batch
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, self.global_step, train_summary_op, self.loss, self.accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            #print("step {}, loss {}, acc {}".format(step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

            return loss

        def dev_step(batch, writer=None):

            statement, topic, speaker, state, party, job, location, ch, statement_sq, topic_sq, location_sq, y_batch \
                = zip(*batch)
            feed_dict = {
                self.statement: statement,
                self.topic: topic,
                self.speaker: speaker,
                self.state: state,
                self.party: party,
                self.job: job,
                self.location: location,
                self.ch: ch,
                self.statement_sq: statement_sq,
                self.topic_sq: topic_sq,
                self.location_sq: location_sq,
                self.dropout_keep_prob: 1,
                self.input_y: y_batch
            }

            step, summaries, loss, accuracy, statement_attend = sess.run(
                [self.global_step, dev_summary_op, self.loss, self.accuracy, self.statement_attend],
                feed_dict)

            # print("step %8d, loss %4.3f, acc %4.3f" % (step, loss, accuracy)
            # if writer:
            #     writer.add_summary(summaries, step)
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
            dev_step(test_data)

def eval(test_data, y_test, batch_size, checkpoint_dir, binary=True):
    #statement, topic, speaker, state, party, job, location, ch, statement_sq, topic_sq, location_sq, y_test = zip(*test_data)
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
            statement = graph.get_operation_by_name("statement").outputs[0]
            topic = graph.get_operation_by_name("topic").outputs[0]
            speaker = graph.get_operation_by_name("speaker").outputs[0]
            state = graph.get_operation_by_name("state").outputs[0]
            party = graph.get_operation_by_name("party").outputs[0]
            job = graph.get_operation_by_name("job").outputs[0]
            location = graph.get_operation_by_name("location").outputs[0]
            ch = graph.get_operation_by_name("credit_history").outputs[0]
            statement_sq = graph.get_operation_by_name("statement_sq").outputs[0]
            topic_sq = graph.get_operation_by_name("topic_sq").outputs[0]
            location_sq = graph.get_operation_by_name("location_sq").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            #batches = data_utils.batch_iter(list(zip(statement, topic, speaker, state, party, job, location, ch, statement_sq, topic_sq, location_sq)), batch_size, shuffle=False)
            batches = data_utils.batch_iter(test_data, batch_size, shuffle= False)
            # Collect the predictions here
            all_predictions = []

            for batch in batches:
                statement_batch, topic_batch, speaker_batch, state_batch, party_batch, \
                job_batch, location_batch, ch_batch, statement_sq_batch, topic_sq_batch, location_sq_batch = zip(*batch)
                feed_dict = {
                    statement: statement_batch,
                    topic: topic_batch,
                    speaker: speaker_batch,
                    state: state_batch,
                    party: party_batch,
                    job: job_batch,
                    location: location_batch,
                    ch: ch_batch,
                    statement_sq: statement_sq_batch,
                    topic_sq: topic_sq_batch,
                    location_sq: location_sq_batch,
                    dropout_keep_prob: 1
                }
                batch_predictions = sess.run(predictions, feed_dict= feed_dict)
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
        print("acc = {:04.3f}, precision= {:04.3f}, recall= {:04.3f}, f1= {:04.3f}".format(acc, precision, recall, f1))
        print("confusion")
        print(confusion)

    return acc, precision, recall, f1