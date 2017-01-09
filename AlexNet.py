# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import math
import numpy as np

#########
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def out_conv(i, p, k, s):
  assert s > 0
  return math.floor((i + 2 * p - k) / s) + 1

def out_pool(i, k, s):
  return out_conv(i, 0, k, s)

def maxpool2d(x, k=2, padding='SAME'):
  # MaxPool2D wrapper
  return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)

def conv2d(x, W, b, strides=1, padding='SAME'):
  # Conv2D wrapper, with bias and relu activation
  x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
  x = tf.nn.bias_add(x, b)
  return tf.nn.relu(x)


# Model.
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
  '''From https://github.com/ethereon/caffe-tensorflow
  '''
  c_i = input.get_shape()[-1]
  assert c_i%group==0
  assert c_o%group==0
  convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
  if group==1:
    conv = convolve(input, kernel)
  else:
    input_groups = tf.split(3, group, input)
    kernel_groups = tf.split(3, group, kernel)
    output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
    conv = tf.concat(3, output_groups)
  return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


def load_data(pickle_file = 'notMNIST.pickle'): 

  with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def doJob(valid_dataset,test_dataset,tf_train_dataset,tf_train_labels,dropout):

  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal([11, 11, num_channels, 96], stddev=0.01))
  layer1_biases = tf.Variable(tf.zeros([96]))

  layer2_weights = tf.Variable(tf.truncated_normal([5, 5, 48, 256], stddev=0.01))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[256]))

  layer3_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[384]))

  layer4_weights = tf.Variable(tf.truncated_normal([3, 3, 192, 384], stddev=0.01))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[384]))

  layer5_weights = tf.Variable(tf.truncated_normal([3, 3, 192, 256], stddev=0.01))
  layer5_biases = tf.Variable(tf.constant(1.0, shape=[256]))

  layer6_weights = tf.Variable(tf.truncated_normal([1 * 1 * 256, num_hidden], stddev=0.01))
  layer6_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

  layer7_weights = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.01))
  layer7_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

  layer8_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.01))
  layer8_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  def model(data, dropout):
    print("\n>>> data:" + str(data.get_shape().as_list()))

    ##############conv1
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1_in = conv(data, layer1_weights, layer1_biases, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)
    print("conv1:" + str(conv1.get_shape().as_list()))

    #lrn1
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,depth_radius=radius,alpha=alpha,beta=beta,bias=bias)

    #maxpool1
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    ##############conv2
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2_in = conv(maxpool1, layer2_weights, layer2_biases, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)
    print("conv2:" + str(conv2.get_shape().as_list()))

    #lrn2
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,depth_radius=radius,alpha=alpha,beta=beta,bias=bias)

    #maxpool2                                                
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    ##############conv3
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3_in = conv(maxpool2, layer3_weights, layer3_biases, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)
    print("conv3:" + str(conv3.get_shape().as_list()))

    ##############conv4
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4_in = conv(conv3, layer4_weights, layer4_biases, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)
    print("conv4:" + str(conv4.get_shape().as_list()))

    ##############conv5
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5_in = conv(conv4, layer5_weights, layer5_biases, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)
    print("conv5:" + str(conv5.get_shape().as_list()))

    #maxpool5
    #k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    #maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    maxpool5 = conv5
    print("maxpool5 --modified--:" + str(maxpool5.get_shape().as_list()))

  
    ##############fc6
    shape = maxpool5.get_shape().as_list()
    reshape = tf.reshape(maxpool5, [shape[0], shape[1] * shape[2] * shape[3]])
    fc6 = tf.nn.relu(tf.matmul(reshape, layer6_weights) + layer6_biases)
    print("fc6:" + str(fc6.get_shape().as_list()))

    if dropout > 0:
      fc6 = tf.nn.dropout(fc6, dropout)
      print("dropout1:" + str(fc6.get_shape().as_list()))

    ##############fc7
    fc7 = tf.nn.relu_layer(fc6, layer7_weights, layer7_biases)
    print("fc7:" + str(fc7.get_shape().as_list()))

    if dropout > 0:
      fc7 = tf.nn.dropout(fc7, dropout)
      print("dropout2:" + str(fc7.get_shape().as_list()))

    ##############fc8
    fc8 = tf.nn.xw_plus_b(fc7, layer8_weights, layer8_biases)
    print("fc8:" + str(fc8.get_shape().as_list()))

    return fc8

  # Training computation.
  logits = model(tf_train_dataset, dropout)
  #logits = fc8
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

  # Optimizer.
  #optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.05, global_step, 100000, 0.96, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 0))
  test_prediction = tf.nn.softmax(model(tf_test_dataset, 0))

  return optimizer, loss, train_prediction, valid_prediction, test_prediction

#########
image_size = 28
num_labels = 10
num_channels = 1  # grayscale
num_steps = 3000001
batch_size = 128
num_hidden = 4096

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_data()

graph = tf.Graph()
with tf.Session(graph=graph) as session:

    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    optimizer, loss, train_prediction, valid_prediction, test_prediction  = doJob(valid_dataset,test_dataset,tf_train_dataset,tf_train_labels,0.5)

   
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 50 == 0):
            print('\nMinibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
            print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
