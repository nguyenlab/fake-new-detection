import numpy as np
import pandas as pd
import re
import collections
import itertools
import numpy as np
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string

LABELs = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
nltk_stopwords = set(stopwords.words('english'))

class LiarDataset(object):
    def __init__(self, file_path):

        df = pd.read_table(file_path, header=None)

        statements = df.iloc[:, 2].tolist()
        self.statement = []

        self.cate_label = df.iloc[:, 1].tolist()
        self.speaker = df.iloc[:, 4].tolist()
        self.speaker = replace_nan(self.speaker)
        self.job = df.iloc[:, 5].tolist()
        self.job = replace_nan(self.job)
        self.state = df.iloc[:, 6].tolist()
        self.state = replace_nan(self.state)
        self.party = df.iloc[:, 7].tolist()
        self.party = replace_nan(self.party)
        location = df.iloc[:, 13].tolist()
        self.location = replace_nan(location)
        # for idx, lc in enumerate(np.array(location)):
        #     if lc == "nan":
        #         self.location.append(lc)
        #     else:
        #         self.location.append(clean_text(lc))

        self.topic = [t.split(",") for t in df.iloc[:, 3].tolist()]
        #self.topic = replace_nan(self.topic)
        barely_true_count = df.iloc[:, 8].tolist()
        false_count = df.iloc[:, 9].tolist()
        half_true_count = df.iloc[:, 10].tolist()
        mostly_true_count = df.iloc[:, 11].tolist()
        pants_fire_count = df.iloc[:, 12].tolist()

        self.credit_history = zip(barely_true_count, false_count, half_true_count,
                                  mostly_true_count, pants_fire_count)

        self.credit_history = np.array(self.credit_history, dtype=np.float32)
        self.credit_history = normalize_ch(self.credit_history)
        # clean text
        for statement in statements:
            self.statement.append(clean_text(statement))


def to_6d_onehot_label(cate_labels):
    labels = []
    for cate_label in cate_labels:
        label_index = LABELs.index(cate_label.lower().strip())
        label = [0] * 6
        label[label_index] = 1
        labels.append(label)

    return labels


def to_2d_onehot_label(cate_labels):
    labels = []
    for cate_label in cate_labels:
        label_index = LABELs.index(cate_label.lower().strip())

        if label_index in [0, 1, 2]:
            label_index = 0
        else:
            label_index = 1

        label = [0] * 2
        label[label_index] = 1
        labels.append(label)

    return labels

def concate_text(texts):
    texts = np.array(texts)
    concated = []

    for idx, text in enumerate(texts):
        text = text.lower().strip().split()
        text = '-'.join(text)
        concated.append(text)

    return concated


def split_text(texts):
    return [text.split("-") for text in texts]

def load_data(file_path):
    dataset = LiarDataset(file_path)
    statement = dataset.statement
    credit_history = dataset.credit_history
    #label = to_6d_onehot_label(dataset.cate_label)
    label = to_2d_onehot_label(dataset.cate_label)
    label = np.array(label)
    topic = dataset.topic
    job = concate_text(dataset.job)
    state = concate_text(dataset.state)
    party = dataset.party
    location = dataset.location  # deal with location later
    speaker = dataset.speaker

    return statement, credit_history, topic, speaker, job, state, party, location, label

def clean_text(string):
    string = re.sub(r"\d+.\d+", "fnumber", string)
    string = re.sub(r"\d+", "fnumber", string)
    # string = re.sub(r"\$\d+", "fmoney", string)
    string = re.sub(r"\$ fnumber", "fmoney", string)
    string = re.sub(r"fnumber %", "f-percent", string)
    string = re.sub(r"fnumber percent", "f-percent", string)
    # string = re.sub(r"\d+ percent", " f-percent", string)
    # string = re.sub(r"\d+", "", string)
    # string = re.sub(r"\.f-percent", "f-percent", string)
    string = re.sub(r"Ive", "I n\'ve", string)
    string = re.sub(r"dont", "do n\'t", string)
    string = re.sub(r"doesnt", "does n\'t", string)
    string = re.sub(r"wont", "will n\'t", string)
    string = re.sub(r"cant", "can n\'t", string)
    string = re.sub(r"wouldnt", "would n\'t", string)
    string = re.sub(r"couldnt", "could n\'t", string)
    string = re.sub(r"shouldnt", "should n\'t", string)
    string = re.sub(r"wasnt", "was n\'t", string)
    string = re.sub(r"werent", "were n\'t", string)
    string = re.sub(r"hasnt", "has n\'t", string)
    string = re.sub(r"havent", "have n\'t", string)
    string = re.sub(r"\"", " \" ", string)
    string = re.sub(r" \'", " \' ", string)
    string = re.sub(r"\' ", " \' ", string)
    string = string.strip().lower()
    final_string = ' '.join([word for word in string.split() if word not in nltk_stopwords])
    return final_string


def create_wordlist(texts):
    """

    :param texts: a list tokenize text, each is a list of tokens 
    :return: 
    """
    wordlist = []
    for text in texts:
        for word in text:
            wordlist.append(word)

    return wordlist


def build_vocab(words):
    count = [['PAD', -1]]
    count.append(['UNK', -1])
    count.extend(collections.Counter(words).most_common())

    word_index = dict()
    for word, _ in count:
        word_index[word] = len(word_index)
    index_word = dict(zip(word_index.values(), word_index.keys()))

    return word_index, index_word


def normalize(vectors):
    """
    Normalize the embeddings to have norm 1.
    :param embeddings: 2-d numpy array
    :return: normalized embeddings
    """
    # normalize embeddings
    norms = np.linalg.norm(vectors, axis=1).reshape((-1, 1))
    # deal with nan
    return vectors / norms


def normalize_ch(vectors):
    normalized_vector = np.zeros_like(vectors, dtype=np.float32)
    unit_vector = np.zeros_like(vectors[0])

    for idx, vector in enumerate(vectors):
        # elements = list(set(vector))
        if not (all(vector == 0) or all(np.isnan(vector))):
            norm = np.linalg.norm(vector)
            normalized_vector[idx, :] = vector / norm

    return normalized_vector


def load_embedding(path):
    embeddings_index = {}
    f = open(path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def load_dep_embedding(path):
    embeddings_index = {}
    f = open(path)
    for line in f:
        values = line.split()
        word = values[0].split("_")[1]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def get_embedding_matrix(word_index, embeddings_index, EMBEDDING_DIM):
    embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM))
    embedding_matrix[0] = generate_random_vector(EMBEDDING_DIM)
    embedding_matrix[1] = generate_random_vector(EMBEDDING_DIM)

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def generate_random_vector(size):
    """
    Generate a random vector from a uniform distribution between
    -0.1 and 0.1.
    """
    return np.random.uniform(-0.1, 0.1, size)

def tokens_to_indices_1d(word_dict, tokens):
    indices = []
    for token in tokens:
        id = word_dict.get(token)
        if id == None:
            id = 1
        indices.append(id)

    return indices


def tokens_to_indices(word_dict, sents, max_len):
    """
    convert a list of tokens representing tokens into a 2D array [sentence_length, max_words] of index
    :param word_dict: {word, index}
    :param tokens: a list, each element is a list of tokens
    :return: list, each element is an index
    """
    index_matrix = np.zeros(shape=[len(sents), max_len], dtype=np.int32)

    for i, sent in enumerate(sents):
        sent_ids = []
        len_sent = len(sent)
        for word in sent:
            index = word_dict.get(word)
            if index is not None:
                sent_ids.append(index)
            else:
                sent_ids.append(1)
        if len_sent <= max_len:
            index_matrix[i, :len_sent] = sent_ids
        else:
            index_matrix[i,] = sent_ids[:max_len]

    return index_matrix


def tokenize(text):
    """
    tokenize a text
    :param text: 
    :return: 
    """
    # remove double quote in text
    text = text.replace('"', '')
    word_list = []
    sent_list = sent_tokenize(text)
    for sent in sent_list:
        tokens = word_tokenize(sent)
        word_list.extend(tokens)

    return word_list


def texts_to_tokens(texts):
    """
    tokenize a list of text
    :param texts: 
    :return: 
    """
    token_list = []
    for text in texts:
        tokens = tokenize(text.decode('utf-8'))
        new_tokens = [token for token in tokens if token not in string.punctuation
                      and token not in ['``', '--', '...']]
        token_list.append(new_tokens)

    return token_list

def batch_iter(data, batch_size, shuffle=True):
    """
    Generates mini-batches for 1 epoch
    Wrong here
    """
    data = np.array(data)
    n = len(data)
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)
        data = data[idx_list]

    num_batches = (n - 1) / batch_size + 1

    for batch_i in range(num_batches):
        start_index = batch_size * batch_i
        end_index = min((batch_i + 1) * batch_size, n)
        yield data[start_index: end_index]


def print_list(list):
    for element in list:
        print(element)


def get_sequence_length(token_list):
    return [len(tokens) for tokens in token_list]


def get_label_statistics(cate_labels, value):
    count = 0
    for label in cate_labels:
        if label == value:
            count += 1
    return count

def convert_ch(chs, size):
    zero = np.zeros(shape= (len(chs), size))
    for idx, ch in enumerate(chs):
        zero[idx,:len(ch)] = ch
    return zero

def replace_nan(texts):

    texts_ = []
    for idx, lc in enumerate(np.array(texts)):
        if lc == "nan":
            texts_.append(lc)
        else:
            texts_.append(clean_text(lc))

    return texts_

def list_to_2d_matrix(l):
    return [[element] for element in l]

# def check_nan(lists):
#     for list in lists:
#         for e in list:
#             if np.isnan(e):
#                 return True
#     return False
#
# def check_nan2(list):
#     for e in list:
#        if np.isnan(e):
#            return True
#     return False

def char2id(char, vocabulary_size, first_letter):

    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0