import string
import numpy as np

def char2id(char):
    first_letter = ord(string.ascii_lowercase[0])
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        #print('Unexpected character: %s' % char)
        return 0

def sent_to_char_id(sents):
    """
    :param sents: list of sentences, where each is a list a list of words
    :return:
    """
    return [[[char2id(ch) for ch in word if ch not in string.punctuation] for word in sent] for sent in sents]

def info_to_char(side_infos):
    return [[char2id(ch) for ch in side] for side in side_infos]

def padding_3d(sents, max_2d, max_3d):
    """
    :param sents: given sents with shape (,,)
    :param max_2d: max_word
    :param max_3d: max_ch
    :return: 
    """
    padded_sents = np.zeros(shape= [len(sents), max_2d, max_3d])

    for i_s, sent in enumerate(sents):
        padded_sent = np.zeros(shape=[max_2d, max_3d])

        for i_w, word in enumerate(sent):
            if i_w < max_2d:
                if len(word) <= max_3d:
                    padded_sent[i_w, :len(word)] = word
                else:
                    padded_sent[i_w, :] = word[:max_3d]

        padded_sents[i_s,:,:] = padded_sent

    return padded_sents

def padding_2d(side_info, max_ch):
    """
    :param side_info: 
    :param max_word: 
    :return: 
    """
    padded_info = np.zeros(shape=[len(side_info), max_ch])

    for idx, side in enumerate(side_info):
        if len(side) <= max_ch:
            padded_info[idx, : len(side)] = side
        else:
            padded_info[idx, : ] = side[:max_ch]

    return padded_info