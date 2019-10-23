import tensorflow as tf
from setting import *

def conv_layer(input, weights, bias, pad, stride):
    pad = pad[0]
    stride = stride[0]
    input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
    w = tf.Variable(weights, name='weight', dtype='float32')
    b = tf.Variable(bias, name='bias', dtype='float32')
    conv = tf.nn.conv2d(input, w, strides=[1, stride[0], stride[1], 1], padding='VALID')
    return tf.nn.bias_add(conv, b)

def full_conv(input, weights, bias):
    w = tf.Variable(weights, name='weight', dtype='float32')
    b = tf.Variable(bias, name='bias', dtype='float32')
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID')
    return tf.nn.bias_add(conv, b)

def pool_layer(input, stride, pad, area):
    # with tf.variable_scope(name):
    pad = pad[0]
    area = area[0]
    stride = stride[0]
    input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
    return tf.nn.max_pool(input, ksize=[1, area[0], area[1], 1], strides=[1, stride[0], stride[1], 1], padding='VALID')


def conv2d(input, kernel, strides, padding, init_rate, collection_name=None, bias=True, trainable=True):
    W = tf.get_variable(name='weight', shape=kernel,
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=init_rate),
                        collections=collection_name, trainable=trainable)
    b = tf.get_variable(name='bias', shape=[kernel[-1]],
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=init_rate),
                        collections=collection_name, trainable=trainable)
    out = tf.nn.conv2d(input, W, strides=strides, padding=padding)
    if bias:
        out = tf.nn.bias_add(out, b)
    return out


# activate function
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

def tanh(x):
    return tf.nn.tanh(x)

# loss
def mse_criterion(in_, target):
    in_ = tf.squeeze(in_)
    target = tf.squeeze(target)
    loss = tf.nn.l2_loss(in_ - target)
    return loss


def sce_criterion(logits, labels):

    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def calc_neighbor(label_1, label_2):
    Sim = (np.dot(label_1, label_2.transpose()) > 0).astype(int)
    return Sim


def local_norm(x):
    return tf.nn.local_response_normalization(x, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)

def batch_norm(x):
    return tf.contrib.layers.batch_norm(x, decay=0.999, updates_collections=None, epsilon=0.001, scale=True) #epsilon=1e-5,

