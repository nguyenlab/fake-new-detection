from data_utils import load_data, texts_to_tokens, tokens_to_indices, build_vocab, load_embedding, get_embedding_matrix, get_sequence_length, print_list, get_label_statistics
from memnet.memnet import MemNet, eval
import numpy as np
import tensorflow as tf
import itertools

tf.flags.DEFINE_integer("num_epochs", 53, "number of epochs") # 125 # 84
tf.flags.DEFINE_integer("batch_size", 64, "size of a batch")
tf.flags.DEFINE_integer("word_embedding_size", 50, "size of word embedding")
tf.flags.DEFINE_integer("attention_size", 32, "attention_size")
tf.flags.DEFINE_float("learning_rate", 0.25, "learning_rate")
tf.flags.DEFINE_float("dropout_keep_prob", .8, "dropout_keep_prob")
tf.flags.DEFINE_integer("num_hops", 1, "the number of hops") # 1
tf.flags.DEFINE_integer("seed", 123, "random seed")

FLAGS  = tf.flags.FLAGS
FLAGS._parse_flags()

MAXLEN = 30
np.random.seed(FLAGS.seed)

train_file = "dataset/train.tsv"
valid_file = "dataset/valid.tsv"
test_file = "dataset/test.tsv"

train_statement, train_ch, train_topic, train_speaker, train_job, train_state,\
train_party, train_location, train_y = load_data(train_file)

topic_list = itertools.chain.from_iterable(train_topic)
topic_index, _ = build_vocab(topic_list)

valid_statement, valid_ch, valid_topic, valid_speaker, valid_job, valid_state,\
valid_party, valid_location, valid_y = load_data(valid_file)

test_statement, test_ch, test_topic, test_speaker, test_job, test_state,\
test_party, test_location, test_y = load_data(test_file)

# get dataset statistics

train_tokens = texts_to_tokens(train_statement)
valid_tokens = texts_to_tokens(valid_statement)
test_tokens = texts_to_tokens(test_statement)

train_sq_len = get_sequence_length(train_tokens)
valid_sq_len = get_sequence_length(valid_tokens)
test_sq_len = get_sequence_length(test_tokens)

# create vocabulary from the data itself
wordlist = itertools.chain.from_iterable(train_tokens)
word_index, _ = build_vocab(wordlist)

vocab_size = len(word_index)

# convert words to indices including padding and cutting
train_x = tokens_to_indices(word_index, train_tokens, MAXLEN)
valid_x = tokens_to_indices(word_index, valid_tokens, MAXLEN)
test_x = tokens_to_indices(word_index, test_tokens, MAXLEN)

train_data = zip(train_x, train_sq_len, train_ch, train_y)
valid_data = zip(valid_x, valid_sq_len, valid_ch, valid_y)
test_data = zip(test_x, test_sq_len, test_ch, test_y)

# Create a model
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:

        mem_net = MemNet(
            vocab_size= vocab_size,
            statment_size= MAXLEN,
            word_embedding_size= FLAGS.word_embedding_size,
            num_hops= FLAGS.num_hops,
            ch_size = train_ch.shape[1],
            num_classes= train_y.shape[1],
            attention_size= FLAGS.attention_size)
        print("Model created!")

        mem_net.train(sess,
                       train_data= train_data,
                       dev_data= valid_data,
                       test_data = test_data,
                       num_epochs= FLAGS.num_epochs,
                       batch_size= FLAGS.batch_size,
                       dropout_keep= FLAGS.dropout_keep_prob,
                       starter_learning_rate= FLAGS.learning_rate
                       )
    
        print("\nFinish training")
        # eval(train_data, batch_size=FLAGS.batch_size, checkpoint_dir=mem_net.checkpoint_dir, binary=False)

        print("\ndev")
        eval(valid_data, batch_size=FLAGS.batch_size, checkpoint_dir=mem_net.checkpoint_dir, binary=False)

        print("\ntest")
        eval(test_data, batch_size= FLAGS.batch_size, checkpoint_dir= mem_net.checkpoint_dir, binary= False)