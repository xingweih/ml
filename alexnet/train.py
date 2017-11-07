#!/home/hxw/anaconda3/envs/tensorflow/bin/python

import tensorflow as tf
import numpy as np 

def fc(x, sizeIn, sizeOut, name, relu=True):
	with tf.variable_scope(name) as scope:
		weights = tf.get_variable('weights', shape=[sizeIn, sizeOut], trainable=True)
		biases = tf.get_variable('biases', shape=[sizeOut], trainable=True)
		act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
		if(relu == True):
			relu = tf.nn.relu(act)
			return relu
		else:
			return act

def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)

def inference(images):
	with tf.variable_scope('conv1') as scope:
		kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96],
			dtype = tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='VALID')
		biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias, name=scope)

	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
		padding='VALID', name='pool1')

	with tf.variable_scope('lrn1') as scope:
		lrn1 = tf.nn.local_response_normalizetion(pool1, alpha=1e-4,
			beta=0.75, depth_radius=2, bias=2.0)
	
	with tf.variable_scope('conv2') as scope:
		kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256],
			dtype = tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias, name=scope)

	pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
		padding='VALID', name='pool2')

	with tf.variable_scope('lrn2') as scope:
		lrn2 = tf.nn.local_response_normalizetion(pool2, alpha=1e-4,
			beta=0.75, depth_radius=2, bias=2.0)

	with tf.variable_scope('conv3') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
			dtype = tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(bias, name=scope)
	
	with tf.variable_scope('conv4') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384],
			dtype = tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(bias, name=scope)

	with tf.variable_scope('conv5') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
			dtype = tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(bias, name=scope)

	pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
		padding='VALID', name='pool2')

	flatten = tf.reshape(pool5, [-1, 6*6*256])
	fc6 = fc(flatten, 6*6*256, 4096, name='fc6')
	dropout6 = dropout(fc6, 0.5)

	fc7 = fc(fc6, 4096, 4096, name='fc7')
	dropout7 = dropout(fc7, 0.5)

	fc8 = fc(fc7, 4096, 1000, name='fc7')
	dropout8 = dropout(fc8, 0.5)

def load_initial_params(weight_bias):
	weights_dict = np.load(weight_bias, encoding='bytes').item()
	print(weights_dict.keys())
	for op_name in weights_dict:
		print(op_name)
		with tf.variable_scope(op_name, reuse=True) as scope:
			for data in weights_dict[op_name]:
				print(scope)
				if(len(data.shape) == 1):
					var = tf.get_variable('biases', trainable=False)
				else:
					var = tf.get_variable('weights', trainable=False)
					
if __name__ == '__main__':
	print('begin')
	load_initial_params('alexnet.npy')
	print('end')
