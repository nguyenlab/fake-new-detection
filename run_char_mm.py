from data_utils2 import load_data, texts_to_tokens, tokens_to_indices, build_vocab, \
    get_sequence_length, tokens_to_indices_1d, clean_text, convert_ch, list_to_2d_matrix, print_list
from char_mm.char_mm import MemNet, eval
import numpy as np
import tensorflow as tf
import itertools
import string
from char_utils import sent_to_char_id, padding_3d, padding_2d, info_to_char

tf.flags.DEFINE_integer("num_epochs", 41, "number of epochs") # 41 for 2-classes
tf.flags.DEFINE_integer("batch_size", 16, "size of a batch")
tf.flags.DEFINE_integer("embedding_size", 20, "size of word embedding") # 50
tf.flags.DEFINE_float("learning_rate", 0.1, "learning_rate")
tf.flags.DEFINE_float("dropout_keep_prob", .8, "dropout_keep_prob")
tf.flags.DEFINE_integer("seed", 123, "random seed")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("hops", 1, "Number of hops in the Memory Network.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

MAXLEN = 15
MAXCHAR = 18 # 18 for speaker
np.random.seed(FLAGS.seed)

train_file = "dataset/train.tsv"
valid_file = "dataset/valid.tsv"
test_file = "dataset/test.tsv"

vocab_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

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

train_char_tokens = sent_to_char_id(train_tokens)
valid_char_tokens = sent_to_char_id(valid_tokens)
test_char_tokens = sent_to_char_id(test_tokens)

train_x = padding_3d(train_char_tokens, MAXLEN, MAXCHAR)
valid_x = padding_3d(valid_char_tokens, MAXLEN, MAXCHAR)
test_x = padding_3d(test_char_tokens, MAXLEN, MAXCHAR)

# pa to vec
train_party_tokens = info_to_char(train_party)
valid_party_tokens = info_to_char(valid_party)
test_party_tokens = info_to_char(test_party)

train_party = padding_2d(train_party_tokens, MAXCHAR)
valid_party = padding_2d(valid_party_tokens, MAXCHAR)
test_party = padding_2d(test_party_tokens, MAXCHAR)

train_state_tokens = info_to_char(train_state)
valid_state_tokens = info_to_char(valid_state)
test_state_tokens = info_to_char(test_state)

train_state = padding_2d(train_state_tokens, MAXCHAR)
valid_state = padding_2d(valid_state_tokens, MAXCHAR)
test_state = padding_2d(test_state_tokens, MAXCHAR)

train_speaker_tokens = info_to_char(train_speaker)
valid_speaker_tokens = info_to_char(valid_speaker)
test_speaker_tokens = info_to_char(test_speaker)

train_speaker = padding_2d(train_speaker_tokens, MAXCHAR)
valid_speaker = padding_2d(valid_speaker_tokens, MAXCHAR)
test_speaker = padding_2d(test_speaker_tokens, MAXCHAR)

# train_data = zip(train_x, train_party, train_y)
# valid_data = zip(valid_x, valid_party, valid_y)
# valid_data2 = zip(valid_x, valid_party)
# test_data = zip(test_x, test_party)

# train_data = zip(train_x, train_state, train_y)
# valid_data = zip(valid_x, valid_state, valid_y)
# valid_data2 = zip(valid_x, valid_state)
# test_data = zip(test_x, test_state)

train_data = zip(train_x, train_speaker, train_y)
valid_data = zip(valid_x, valid_speaker, valid_y)
valid_data2 = zip(valid_x, valid_speaker)
test_data = zip(test_x, test_speaker)

# Create a model
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        model = MemNet(vocab_size= vocab_size,
                       char_size= MAXCHAR,
                       statement_size= MAXLEN,
                       embedding_size= FLAGS.embedding_size,
                       num_hops=FLAGS.hops,
                       num_classes= len(train_y[0]))
        print("Model created!")
        print("Begin training!")
        model.train(sess,
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
        eval(valid_data2, valid_y, batch_size=FLAGS.batch_size, checkpoint_dir=model.checkpoint_dir, binary=False)

        print("\ntest")
        eval(test_data, test_y, batch_size=FLAGS.batch_size, checkpoint_dir=model.checkpoint_dir, binary=False)