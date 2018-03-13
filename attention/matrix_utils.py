import tensorflow as tf
import numpy as np


def base_attention(A, v, max_len, hidden_size, sq):
    
    if sq != None:
        A_3dmask = add_3dmask(A, sq, max_len, hidden_size, mask_value= 0)
    else:
        A_3dmask = A
    
    p = matrix_vector_mul(A_3dmask, v)
    if sq != None:
        p = add_2dmask(p, sq, max_len, mask_value= -np.inf)
        p = tf.nn.softmax(p)
    h = weighted_sum(A_3dmask, p)

    # if sq != None:
    #     return h, A_3dmask, p

    return h

def matrix_vector_mul(A, v):
    """ Perform v^T*A
    :param A: 3-d matrix with shape (?, len, dim) 
    :param v: 2-d vector with shape (?, dim)
    :return: 2-d matrix with shape (?, dim)
    """
    v_temp = tf.transpose(tf.expand_dims(v, axis= -1), [0, 2, 1])
    sum = tf.reduce_sum(A*v_temp, axis= 2)

    return sum

def weighted_sum(C, v):
    C_temp = tf.transpose(C, [0, 2, 1]) # (?, dim, len)
    v_temp = tf.transpose(tf.expand_dims(v, -1), (0, 2, 1)) # (?, 1, len) --> (?, dim, len)
    sum = tf.reduce_sum(C_temp*v_temp, axis= 2)

    return sum

def add_3dmask(A, sq, max_len, hidden_size, mask_value= 0):

    mask = tf.sequence_mask(lengths=sq, maxlen= max_len)
    mask3d = tf.expand_dims(mask, 1)
    mask3d = tf.tile(mask3d, (1, hidden_size, 1))
    mask3d = tf.transpose(mask3d, (0, 2, 1))

    ones = tf.ones_like(A, dtype=tf.int32)
    pad_values = mask_value * tf.cast(ones, tf.float32)
    A_mask = tf.where(mask3d, A, pad_values)

    return A_mask

def add_2dmask(A, sq, max_len, mask_value= -np.inf):

    mask = tf.sequence_mask(lengths=sq, maxlen= max_len)
    ones = tf.ones_like(A, dtype=tf.int32)
    pad_values = mask_value * tf.cast(ones, tf.float32)
    A_mask = tf.where(mask, A, pad_values)

    return A_mask


def attend_concate(h, u, sequence_length, H, U, A, reuse, name):
    """
    perform a general type of attention based on agregation
    :param h: of shape (B, T, H), to multiply with (H, A) 
    :param u: of shape (B, U) to multiply with (U, A)
    :return: v_t*tanh(h*W_h + u*W_u + b)
    """
    T = h.shape[1]
    with tf.variable_scope(name, reuse= reuse): # change to tf.get_variable()
        initializer = tf.random_normal_initializer(0.0, 0.1)
        #W_h = tf.Variable(tf.truncated_normal(shape=[H, A], mean=0, stddev=0.1), name="W_h")
        #W_u = tf.Variable(tf.truncated_normal(shape=[U, A], mean=0, stddev=0.1), name="W_u")
        W_h = tf.get_variable(name= "W_h",
                        shape= [H, A],
                        dtype= tf.float32,
                        initializer= initializer)

        W_u = tf.get_variable(name="W_u",
                              shape=[U, A],
                              dtype=tf.float32,
                              initializer= initializer)
        #b = tf.get_variable(name= "b")
        b = tf.Variable(tf.constant(0.1, shape=[A], dtype=tf.float32), name= "b")

        agregate = tf.tanh(tf.tensordot(h, W_h, axes=1) + tf.expand_dims((tf.matmul(u, W_u) + b), 1))
        v = tf.Variable(tf.truncated_normal(shape=[A], mean=0, stddev=0.1))
        raw_score = tf.tensordot(agregate, v, axes=1)
        masked_attend = add_mask(raw_score, sequence_length, T)
        p = tf.nn.softmax(masked_attend)

    return p