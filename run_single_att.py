from data_utils2 import load_data, texts_to_tokens, tokens_to_indices, build_vocab, load_embedding, \
    get_embedding_matrix, \
    get_sequence_length, tokens_to_indices_1d, clean_text, convert_ch, list_to_2d_matrix
from attention.single_att import SingAtt, eval
import numpy as np
import tensorflow as tf
import itertools

tf.flags.DEFINE_integer("num_epochs", 31, "number of epochs") # 61
tf.flags.DEFINE_integer("batch_size", 64, "size of a batch")
tf.flags.DEFINE_integer("word_embedding_size", 50, "size of word embedding")
tf.flags.DEFINE_float("learning_rate", 0.5, "learning_rate")
tf.flags.DEFINE_float("dropout_keep_prob", .8, "dropout_keep_prob")
tf.flags.DEFINE_integer("seed", 123, "random seed")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

MAXLEN = 30
np.random.seed(FLAGS.seed)

train_file = "dataset/train.tsv"
valid_file = "dataset/valid.tsv"
test_file = "dataset/test.tsv"

# statement, credit_history, topic, speaker, job, state, party, location, label

train_statement, train_ch, train_topic, train_speaker, train_job, train_state, \
train_party, train_location, train_y = load_data(train_file)

valid_statement, valid_ch, valid_topic, valid_speaker, valid_job, valid_state, \
valid_party, valid_location, valid_y = load_data(valid_file)

test_statement, test_ch, test_topic, test_speaker, test_job, test_state, \
test_party, test_location, test_y = load_data(test_file)

train_location = [clean_text(lc) for lc in train_location]
valid_location = [clean_text(lc) for lc in valid_location]
test_location = [clean_text(lc) for lc in test_location]

train_location = texts_to_tokens(train_location)
valid_location = texts_to_tokens(valid_location)
test_location = texts_to_tokens(test_location)

# text
train_tokens = texts_to_tokens(train_statement)
valid_tokens = texts_to_tokens(valid_statement)
test_tokens = texts_to_tokens(test_statement)

# text sequence
train_sq_len = get_sequence_length(train_tokens)
valid_sq_len = get_sequence_length(valid_tokens)
test_sq_len = get_sequence_length(test_tokens)

# change train_speaker, job, party
speaker_2d = list_to_2d_matrix(train_speaker) + list_to_2d_matrix(valid_speaker) + \
             list_to_2d_matrix(test_speaker)

job_2d = list_to_2d_matrix(train_job) + list_to_2d_matrix(valid_job) + \
         list_to_2d_matrix(test_job)

party_2d = list_to_2d_matrix(train_party) + list_to_2d_matrix(valid_party) + \
           list_to_2d_matrix(test_party)

# create word vocabulary from the statement,
words = train_tokens + train_topic + valid_tokens + valid_topic + test_tokens + test_topic + \
        train_location + valid_location + test_location + speaker_2d + job_2d + party_2d

wordlist = itertools.chain.from_iterable(words)
word_index, _ = build_vocab(wordlist)
vocab_size = len(word_index)

# convert words to indices including padding and cutting
train_x = tokens_to_indices(word_index, train_tokens, MAXLEN)
valid_x = tokens_to_indices(word_index, valid_tokens, MAXLEN)
test_x = tokens_to_indices(word_index, test_tokens, MAXLEN)

# get topic sequence
max_topic = 5
train_topic = tokens_to_indices(word_index, train_topic, max_topic)
valid_topic = tokens_to_indices(word_index, valid_topic, max_topic)
test_topic = tokens_to_indices(word_index, test_topic, max_topic)

# get topic sequence
train_tp_sq = np.array([np.count_nonzero(t) for t in train_topic])
valid_tp_sq = np.array([np.count_nonzero(t) for t in valid_topic])
test_tp_sq = np.array([np.count_nonzero(t) for t in test_topic])

# speaker
train_speaker = tokens_to_indices_1d(word_index, train_speaker)
valid_speaker = tokens_to_indices_1d(word_index, valid_speaker)
test_speaker = tokens_to_indices_1d(word_index, test_speaker)

train_sp_sq = np.array([np.count_nonzero(t) for t in train_topic])
valid_sp_sq = np.array([np.count_nonzero(t) for t in valid_topic])
test_sp_sq = np.array([np.count_nonzero(t) for t in test_topic])

# which side information to use
# job

train_job = tokens_to_indices_1d(word_index, train_job)
valid_job = tokens_to_indices_1d(word_index, valid_job)
test_job = tokens_to_indices_1d(word_index, test_job)
# state

train_state = tokens_to_indices_1d(word_index, train_state)
valid_state = tokens_to_indices_1d(word_index, valid_state)
test_state = tokens_to_indices_1d(word_index, test_state)
# party

train_party = tokens_to_indices_1d(word_index, train_party)
valid_party = tokens_to_indices_1d(word_index, valid_party)
test_party = tokens_to_indices_1d(word_index, test_party)

# train_location, valid_location, test_location
train_location = tokens_to_indices(word_index, train_location, 6)
valid_location = tokens_to_indices(word_index, valid_location, 6)
test_location = tokens_to_indices(word_index, test_location, 6)

train_lc_sq = np.array([np.count_nonzero(t) for t in train_location])
valid_lc_sq = np.array([np.count_nonzero(t) for t in valid_location])
test_lc_sq = np.array([np.count_nonzero(t) for t in test_location])

train_ch = convert_ch(train_ch, FLAGS.word_embedding_size)
valid_ch = convert_ch(valid_ch, FLAGS.word_embedding_size)
test_ch = convert_ch(test_ch, FLAGS.word_embedding_size)

# statement, topic, speaker, state, party, job, location, ch, statement_sq, topic_sq, location_sq, y_batch
train_data = zip(train_x, train_topic, train_speaker, train_state, train_party, train_job,
                 train_location, train_ch, train_sq_len, train_tp_sq, train_lc_sq, train_y)

valid_data = zip(valid_x, valid_topic, valid_speaker, valid_state, valid_party, valid_job,
                 valid_location, valid_ch, valid_sq_len, valid_tp_sq, valid_lc_sq, valid_y)

valid_data2 = zip(valid_x, valid_topic, valid_speaker, valid_state, valid_party, valid_job,
                  valid_location, valid_ch, valid_sq_len, valid_tp_sq, valid_lc_sq)

test_data = zip(test_x, test_topic, test_speaker, test_state, test_party, test_job,
                test_location, test_ch, test_sq_len, test_tp_sq, test_lc_sq)

test_data2 = zip(test_x, test_topic, test_speaker, test_state, test_party, test_job,
                 test_location, test_ch, test_sq_len, test_tp_sq, test_lc_sq, test_y)

# Create a model
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        mem_net = SingAtt(
            vocab_size=vocab_size,
            statment_size=MAXLEN,
            word_embedding_size=FLAGS.word_embedding_size,
            topic_size=len(train_topic[0]),
            location_size=len(train_location[0]),
            side_size=FLAGS.word_embedding_size,
            num_classes=len(train_y[0])
        )
        print("Model created!")
        mem_net.train(sess,
                      train_data=train_data,
                      dev_data=valid_data,
                      test_data=test_data2,
                      num_epochs=FLAGS.num_epochs,
                      batch_size=FLAGS.batch_size,
                      dropout_keep=FLAGS.dropout_keep_prob,
                      starter_learning_rate=FLAGS.learning_rate
                      )
        print("Finish training")
        # print("\ntrain")
        # eval(train_data, batch_size=FLAGS.batch_size, checkpoint_dir=mem_net.checkpoint_dir, binary=False)

        print("\ndev")
        eval(valid_data2, valid_y, batch_size=FLAGS.batch_size, checkpoint_dir=mem_net.checkpoint_dir, binary=False)

        print("\ntest")
        eval(test_data, test_y, batch_size=FLAGS.batch_size, checkpoint_dir=mem_net.checkpoint_dir, binary=False)