import scipy.misc
import scipy.io
from ops import *
from setting import *

def img_net(inputs, bit, numClass, name):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        data = scipy.io.loadmat(Param['MODEL_DIR'])
        layers = (
            'conv1', 'relu1', 'norm1', 'pool1',
            'conv2', 'relu2', 'norm2', 'pool2',
            'conv3', 'relu3',
            'conv4', 'relu4',
            'conv5', 'relu5', 'pool5',
            'fc6', 'relu6',
            'fc7', 'relu7')
        weights = data['layers'][0]

        imgnet = {}
        current = tf.convert_to_tensor(inputs, dtype='float32')
        for i, name in enumerate(layers):
            with tf.variable_scope(name):
                if name.startswith('conv'):
                    kernels, bias = weights[i][0][0][0][0]
                    bias = bias.reshape(-1)
                    pad = weights[i][0][0][1]
                    stride = weights[i][0][0][4]
                    current = conv_layer(current, kernels, bias, pad, stride)
                elif name.startswith('relu'):
                    current = tf.nn.relu(current)
                elif name.startswith('pool'):
                    stride = weights[i][0][0][1]
                    pad = weights[i][0][0][2]
                    area = weights[i][0][0][5]
                    current = pool_layer(current, stride, pad, area)
                elif name.startswith('fc'):
                    kernels, bias = weights[i][0][0][0][0]
                    bias = bias.reshape(-1)
                    current = full_conv(current, kernels, bias)
                elif name.startswith('norm'):
                    current = tf.nn.local_response_normalization(current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
                imgnet[name] = current

        with tf.variable_scope("feature"):
            imgnet['feature'] = batch_norm(
                conv2d(input=current, kernel=[1, 1, current.get_shape()[-1], Param['SEMANTIC_EMBED']],
                                strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0))
            relu8 = relu(imgnet['feature'])

        with tf.variable_scope("hash"):
            imgnet['hash'] = tanh(batch_norm(
                conv2d(input=relu8, kernel=[1, 1, relu8.get_shape()[-1], bit],
                                strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0)))

        with tf.variable_scope("label"):
            imgnet['label'] = sigmoid(batch_norm(
                conv2d(input=relu8, kernel=[1, 1, relu8.get_shape()[-1], numClass],
                                strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0, bias=False)))

        return tf.squeeze(imgnet['hash']), imgnet['feature'], tf.squeeze(imgnet['label'])

def lab_net(label_input, bit, numClass, name):

    with tf.variable_scope(name):

        labnet = {}
        with tf.variable_scope("fc1"):
            relu1 = relu(batch_norm(
                conv2d(input=label_input, kernel=[1, 1, label_input.get_shape()[-1], 4096],
                                strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0)))

        with tf.variable_scope("feature"):
            labnet['feature'] = batch_norm(
                conv2d(input=relu1, kernel=[1, 1, relu1.get_shape()[-1], Param['SEMANTIC_EMBED']],
                                strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0))
            relu2 = relu(labnet['feature'])

        with tf.variable_scope("hash"):
            labnet['hash'] = tanh(batch_norm(
                conv2d(input=relu2, kernel=[1, 1, relu2.get_shape()[-1], bit],
                                strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0)))

        with tf.variable_scope("label"):
            labnet['label'] = sigmoid(batch_norm(
                conv2d(input=relu2, kernel=[1, 1, relu2.get_shape()[-1], numClass],
                                strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0, bias=False)))

        return tf.squeeze(labnet['hash']), labnet['feature'], tf.squeeze(labnet['label'])


def dis_net_IL(feature, name):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        disnet = {}
        with tf.variable_scope("fc1"):
            relu1 = relu(batch_norm(
                conv2d(feature, kernel=[1, 1, feature.get_shape()[-1], 512],
                       strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0)))

        with tf.variable_scope("fc2"):
            relu2 = relu(batch_norm(
                conv2d(relu1, kernel=[1, 1, relu1.get_shape()[-1], 256],
                       strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0)))

        with tf.variable_scope("fc3"):
            disnet['output'] = sigmoid(batch_norm(
                conv2d(relu2, kernel=[1, 1, relu2.get_shape()[-1], 1],
                       strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0)))

        return tf.squeeze(disnet['output'])

def dis_net_TL(feature, name):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        disnet = {}

        with tf.variable_scope("fc1"):
            relu1 = relu(batch_norm(
                conv2d(feature, kernel=[1, 1, feature.get_shape()[-1], 512],
                       strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0)))

        with tf.variable_scope("fc2"):
            relu2 = relu(batch_norm(
                conv2d(relu1, kernel=[1, 1, relu1.get_shape()[-1], 256],
                       strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0)))

        with tf.variable_scope("fc3"):
            disnet['output'] = sigmoid(batch_norm(
                conv2d(relu2, kernel=[1, 1, relu2.get_shape()[-1], 1],
                       strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0)))

        return tf.squeeze(disnet['output'])

def txt_net(text_input, bit, numClass, name):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        txtnet={}
        with tf.variable_scope("MultiScalGeneration"):
            MultiScal = MultiScaleTxt(text_input)

        with tf.variable_scope("MultiScalFusion"):
            relu1 = relu(batch_norm(
                conv2d(MultiScal, kernel=[1, Param['dimText'], 6, 4096],
                                strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0)))

        with tf.variable_scope("feature"):
            txtnet['feature'] = batch_norm(
                conv2d(relu1, kernel=[1, 1, relu1.get_shape()[-1], Param['SEMANTIC_EMBED']],
                       strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0))
            relu2 = relu(txtnet['feature'])

        with tf.variable_scope("hash"):
            txtnet['hash'] = tanh(batch_norm(
                        conv2d(relu2, kernel=[1, 1, relu2.get_shape()[-1], bit],
                               strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0)))

        with tf.variable_scope("label"):
            txtnet['label'] = sigmoid(batch_norm(
                conv2d(relu2, kernel=[1, 1, relu2.get_shape()[-1], numClass],
                       strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0)))

        return tf.squeeze(txtnet['hash']), txtnet['feature'], tf.squeeze(txtnet['label'])

def interp_block(text_input, level):
    with tf.variable_scope('block_'+str(level)):
        shape = [1, 1, 5 * level, 1]
        stride = [1, 1, 5 * level, 1]
        prev_layer = tf.nn.avg_pool(text_input, ksize=shape, strides=stride, padding='VALID')

        prev_layer = relu(batch_norm(
            conv2d(prev_layer, kernel=[1, 1, 1, 1],
                        strides=[1, 1, 1, 1], padding='VALID', init_rate=1.0, bias=False)))

        prev_layer = tf.image.resize_images(prev_layer, [1, Param['dimText']])
        return prev_layer

def MultiScaleTxt(input):
    interp_block1  = interp_block(input, 10)
    interp_block2  = interp_block(input, 6)
    interp_block3  = interp_block(input, 3)
    interp_block6  = interp_block(input, 2)
    interp_block10 = interp_block(input, 1)

    output = tf.concat([input,
                        interp_block10,
                        interp_block6,
                        interp_block3,
                        interp_block2,
                        interp_block1], axis = -1)
    return output
