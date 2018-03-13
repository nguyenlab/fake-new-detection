import tensorflow as tf
import numpy as np

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

def _weight_vector(c, p):
    """ weight vector c by a probability vector p
    :param c: vector to be weighted
    :param p: attention score
    :return: 
    """
    vector_size = c.shape[2]
    p = tf.expand_dims(p, 1)
    weighted_vector = tf.matmul(p, c)
    weighted_vector = tf.reshape(weighted_vector, shape=tf.stack([-1, vector_size]))

    return weighted_vector

def add_mask(h, sequence_lenth, max_sequence, pad_value= 0):
    ones = tf.ones_like(h, dtype=tf.float32)
    pad_values = pad_value * ones

    mask2d = tf.sequence_mask(lengths= sequence_lenth, maxlen= max_sequence) # act as condition
    h_mask = tf.where(mask2d, h, pad_values)
    return mask2d, h_mask

def _attend_concate(h, u, H, U, A):
    """
    perform a general type of attention based on concatenation
    :param h: of shape (B, T, H), to multiply with (H, A) 
    :param u: of shape (B, U) to multiply with (U, A)
    :return: v_t*tanh(h*W_h + u*W_u + b), then perform softmax to get the score
    """
    with tf.name_scope("attend"):
        W_h = tf.Variable(tf.truncated_normal(shape= [H, A], mean= 0, stddev= 0.1), name= "W_h")
        W_u = tf.Variable(tf.truncated_normal(shape= [U, A], mean= 0, stddev= 0.1), name= "W_u")
        b = tf.Variable(tf.zeros(shape= [A], dtype= tf.float32))

        agregate =  tf.tanh(tf.tensordot(h, W_h, axes= 1) + tf.expand_dims((tf.matmul(u, W_u) + b), 1))
        v = tf.Variable(tf.truncated_normal(shape=[A], mean= 0, stddev=0.1))
        raw_attend = tf.tensordot(agregate, v, axes= 1)
    return raw_attend

# create fake data

B = 2
T = 3
H = 4
A = 5
U = 6
sequence_lenth = [3,2]

h = tf.Variable(tf.constant(np.random.randint(2, size= (B, T, H)), dtype= tf.float32))
u = tf.Variable(tf.constant(np.random.randint(2, size= (B, U)), dtype= tf.float32))

raw_attend = _attend_concate(h, u, H, U, A)

mask2d, mask_attend = add_mask(raw_attend, sequence_lenth, T)
p = tf.nn.softmax(mask_attend)
p = add_mask(p, sequence_lenth, T)

result = tf.contrib.learn.run_n(
    {"mask_attend": mask_attend, "p": p, "mask2d":mask2d},
    n= 1,
    feed_dict= None)

print("mask_attend")
print(result[0]["mask_attend"])
# print("p")
# print(result[0]["p"])
print(result[0]["mask2d"])