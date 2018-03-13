from data_utils2 import load_data, texts_to_tokens, tokens_to_indices, build_vocab, load_embedding, get_embedding_matrix, \
    get_sequence_length, print_list, tokens_to_indices_1d, clean_text
from mem_n2n.mem_n2n import MemNet, eval
import numpy as np
import tensorflow as tf
import itertools

tf.flags.DEFINE_integer("num_epochs", 200, "number of epochs")  # 42
tf.flags.DEFINE_integer("batch_size", 64, "size of a batch")
tf.flags.DEFINE_integer("word_embedding_size", 200, "size of word embedding")  # 50
tf.flags.DEFINE_float("learning_rate", 0.5, "learning_rate")
tf.flags.DEFINE_float("dropout_keep_prob", .8, "dropout_keep_prob")
tf.flags.DEFINE_integer("num_hops", 3, "the number of hops")  # 1
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

# create word vocabulary from the statement,
words = train_tokens + train_topic + train_speaker +\
        valid_tokens + valid_topic + valid_speaker +\
        test_tokens + test_topic + test_speaker + \
        train_location + valid_location + test_location

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
# speaker_index, _ = build_vocab(train_speaker)
train_speaker = tokens_to_indices(word_index, train_speaker, 4)
valid_speaker = tokens_to_indices(word_index, valid_speaker, 4)
test_speaker = tokens_to_indices(word_index, test_speaker, 4)

train_sp_sq = np.array([np.count_nonzero(t) for t in train_topic])
valid_sp_sq = np.array([np.count_nonzero(t) for t in valid_topic])
test_sp_sq = np.array([np.count_nonzero(t) for t in test_topic])

#train_location, valid_location, test_location
train_location = [clean_text(lc) for lc in train_location]
valid_location = [clean_text(lc) for lc in valid_location]
test_location = [clean_text(lc) for lc in test_location]

train_location = texts_to_tokens(train_location)
valid_location = texts_to_tokens(valid_location)
test_location = texts_to_tokens(test_location)

train_location = tokens_to_indices(word_index, train_location, 6)
valid_location = tokens_to_indices(word_index, valid_location, 6)
test_location = tokens_to_indices(word_index, test_location, 6)

train_lc_sq = np.array([np.count_nonzero(t) for t in train_location])
valid_lc_sq = np.array([np.count_nonzero(t) for t in valid_location])
test_lc_sq = np.array([np.count_nonzero(t) for t in test_location])

# job
job_index, _ = build_vocab(train_job)
train_job = tokens_to_indices_1d(job_index, train_job)
valid_job = tokens_to_indices_1d(job_index, valid_job)
test_job = tokens_to_indices_1d(job_index, test_job)

# state
state_index, _ = build_vocab(train_state)
train_state = tokens_to_indices_1d(state_index, train_state)
valid_state = tokens_to_indices_1d(state_index, valid_state)
test_state = tokens_to_indices_1d(state_index, test_state)

# party
party_index, _ = build_vocab(train_party)
train_party = tokens_to_indices_1d(party_index, train_party)
valid_party = tokens_to_indices_1d(party_index, valid_party)
test_party = tokens_to_indices_1d(party_index, test_party)

# which side information to use

train_side = train_location
valid_side = valid_location
test_side = test_location

# train_side = train_speaker
# valid_side = valid_speaker
# test_side = test_speaker

# side_index = job_index
# train_side = train_job
# valid_side = valid_job
# test_side = test_job

# train_side = train_topic
# valid_side = valid_topic
# test_side = test_topic

# side_index = party_index
# train_side = train_party
# valid_side = valid_party
# test_side = test_party

# compress all data
# train_data = zip(train_x, train_sq_len, train_side, train_tp_sq, train_y)
# valid_data = zip(valid_x, valid_sq_len, valid_side, valid_tp_sq, valid_y)
# test_data = zip(test_x, test_sq_len, test_side, test_tp_sq, test_y)

train_data = zip(train_x, train_sq_len, train_side, train_lc_sq, train_y)
valid_data = zip(valid_x, valid_sq_len, valid_side, valid_lc_sq, valid_y)
test_data = zip(test_x, test_sq_len, test_side, test_lc_sq, test_y)

# Create a model
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        mem_net = MemNet(
            vocab_size=vocab_size,
            statment_size=MAXLEN,
            word_embedding_size=FLAGS.word_embedding_size,
            num_hops=FLAGS.num_hops,
            num_classes=train_y.shape[1],
            side_size= train_side.shape[1]
            )
        print("Model created!")
        mem_net.train(sess,
                      train_data=train_data,
                      dev_data=valid_data,
                      test_data=test_data,
                      num_epochs=FLAGS.num_epochs,
                      batch_size=FLAGS.batch_size,
                      dropout_keep=FLAGS.dropout_keep_prob,
                      starter_learning_rate=FLAGS.learning_rate
                      )
        print("Finish training")
        # print("\ntrain")
        # eval(train_data, batch_size=FLAGS.batch_size, checkpoint_dir=mem_net.checkpoint_dir, binary=False)

        print("\ndev")
        eval(valid_data, batch_size=FLAGS.batch_size, checkpoint_dir=mem_net.checkpoint_dir, binary=False)

        print("\ntest")
        eval(test_data, batch_size=FLAGS.batch_size, checkpoint_dir=mem_net.checkpoint_dir, binary=False)