#!/home/hxw/anaconda3/envs/tensorflow/bin/python

import tensorflow as tf
import numpy as np 
import person_input

def fc(x, sizeIn, sizeOut, name, relu=True, weight_bias=None):
	with tf.variable_scope(name) as scope:
		if(weight_bias == None):
			weights = tf.get_variable('weights', shape=[sizeIn, sizeOut], trainable=True)
			biases = tf.get_variable('biases', shape=[sizeOut], trainable=True)
		else:
			weights = weight_bias[name][0]
			biases = weight_bias[name][1]
		act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
		if(relu == True):
			relu = tf.nn.relu(act)
			return relu
		else:
			return act

def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)

def inference(images, weight_bias=None, dropoutRate=1.0):
	with tf.variable_scope('conv1') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(images, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv2') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 64, 64],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(conv1, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)
	
	pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], 
						   strides=[1, 2, 2, 1], 
						   padding='VALID', name='pool1')

	with tf.variable_scope('conv3') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv4') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 128, 128],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(conv3, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(pre_activation, name=scope.name)

	pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], 
						   strides=[1, 2, 2, 1], 
						   padding='VALID', name='pool2')

	with tf.variable_scope('conv5') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(pool2, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv6') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(conv5, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv6 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv7') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(conv6, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv7 = tf.nn.relu(pre_activation, name=scope.name)

	pool3 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1], 
						   strides=[1, 2, 2, 1], 
						   padding='VALID', name='pool3')

	with tf.variable_scope('conv8') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 256, 512],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(pool3, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv8 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv9') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(conv8, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv9 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv10') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(conv9, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv10 = tf.nn.relu(pre_activation, name=scope.name)

	pool4 = tf.nn.max_pool(conv10, ksize=[1, 2, 2, 1], 
						   strides=[1, 2, 2, 1], 
						   padding='SAME', name='pool4')

	with tf.variable_scope('conv11') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(pool4, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv11 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv12') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(conv11, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv12 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv13') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
				dtype=tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(conv12, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv13 = tf.nn.relu(pre_activation, name=scope.name)

	pool5 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], 
						   strides=[1, 2, 2, 1], 
						   padding='VALID', name='pool5')

	flatten = tf.reshape(pool5, [-1, 7*7*512])
	fc6 = fc(flatten, 7*7*512, 4096, name='fc6', weight_bias=weight_bias)
		
	dropout6 = dropout(fc6, dropoutRate)

	fc7 = fc(fc6, 4096, 4096, name='fc7', weight_bias=weight_bias)
	dropout7 = dropout(fc7, dropoutRate)

	fc8 = fc(fc7, 4096, 1000, name='fc8', weight_bias=weight_bias)
	dropout8 = dropout(fc8, dropoutRate)
	return dropout8

if __name__ == '__main__':
	print('---------------Program Begin--------------------')
	imageBatch, labelBatch = person_input.input('person_train.tfrecords')
	init = tf.global_variables_initializer()
	with tf.Session() as sess:

		image_raw = tf.gfile.FastGFile('pic.jpg', 'rb').read()
		image = tf.image.decode_jpeg(image_raw, channels=3)
		image = tf.image.resize_images(image, size=(224, 224))
		image = tf.expand_dims(image, 0)

		coord = tf.train.Coordinator()
#		sess.run(tf.global_variables_initializer())
		threads = tf.train.start_queue_runners(coord=coord)
		images, labels = sess.run([imageBatch, labelBatch])

		print(images)
		inference = inference(image)
		res = sess.run([init, inference])

		coord.request_stop()
		coord.join(threads)
	print('---------------Program End--------------------')
	
