#!/home/hxw/anaconda3/envs/tensorflow/bin/python

import tensorflow as tf
import numpy as np 

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

def inference(images, weight_bias, dropoutRate):
	with tf.variable_scope('conv1') as scope:
		if(weight_bias == None):
			kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96],
				dtype = tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), name='biases')
		else:
			kernel = weight_bias['conv1'][0]
			biases = weight_bias['conv1'][1]
		conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias, name='conv1')

	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
		padding='VALID', name='pool1')

	with tf.variable_scope('lrn1') as scope:
		lrn1 = tf.nn.local_response_normalization(pool1, alpha=1e-4,
			beta=0.75, depth_radius=2, bias=2.0)
	
	with tf.variable_scope('conv2') as scope:
		if(weight_bias == None):
			kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256],
				dtype = tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name='biases')
		else:
			kernel = weight_bias['conv2'][0]
			#kernel = tf.concat(2, [kernelHalf.tolist(), kernelHalf.tolist()])
			kernel = np.concatenate((kernel, kernel), axis=2)
			biases = weight_bias['conv2'][1]
		conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias, name='conv2')

	pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
		padding='VALID', name='pool2')

	with tf.variable_scope('lrn2') as scope:
		lrn2 = tf.nn.local_response_normalization(pool2, alpha=1e-4,
			beta=0.75, depth_radius=2, bias=2.0)

	with tf.variable_scope('conv3') as scope:
		if(weight_bias == None):
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
				dtype = tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), name='biases')
		else:
			kernel = weight_bias['conv3'][0]
			biases = weight_bias['conv3'][1]
		conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
		bias = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(bias, name='conv3')
	
	with tf.variable_scope('conv4') as scope:
		if(weight_bias == None):
			kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384],
				dtype = tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), name='biases')
		else:
			kernel = weight_bias['conv4'][0]
			kernel = np.concatenate((kernel, kernel), axis=2)
			biases = weight_bias['conv4'][1]
		conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
		bias = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(bias, name='conv4')

	with tf.variable_scope('conv5') as scope:
		if(weight_bias == None):
			kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
				dtype = tf.float32, stddev=1e-1), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name='biases')
		else:
			kernel = weight_bias['conv5'][0]
			kernel = np.concatenate((kernel, kernel), axis=2)
			biases = weight_bias['conv5'][1]
		conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
		bias = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(bias, name='conv5')

	pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
		padding='VALID', name='pool2')

	flatten = tf.reshape(pool5, [-1, 6*6*256])
	fc6 = fc(flatten, 6*6*256, 4096, name='fc6', weight_bias=weight_bias)
		
	dropout6 = dropout(fc6, dropoutRate)

	fc7 = fc(fc6, 4096, 4096, name='fc7', weight_bias=weight_bias)
	dropout7 = dropout(fc7, dropoutRate)

	fc8 = fc(fc7, 4096, 1000, name='fc8', weight_bias=weight_bias)
	dropout8 = dropout(fc8, dropoutRate)
	return dropout8

def load_initial_params(weight_bias):
	weights_dict = np.load(weight_bias, encoding='bytes').item()
	#print(weights_dict.keys())
	#w1 = tf.Variable(weights_dict['conv1'][0])
	for op_name in weights_dict:
		print(op_name)
		for data in weights_dict[op_name]:
			print(data.shape)
	'''
				if(len(data.shape) == 1):
					var = tf.get_variable(name=op_name, trainable=False)
				else:
					var = tf.get_variable(name=op_name, trainable=False)
	'''
	return weights_dict
					
def test():
	a = tf.constant([1])
	b = tf.constant([2])
	c = tf.add(a, b)
	return c

if __name__ == '__main__':
	print('begin')
	with tf.Session() as sess:
		#tf.global_variables_initializer
		image_raw = tf.gfile.FastGFile('pic.jpg', 'rb').read()
		image = tf.image.decode_jpeg(image_raw, channels=3)
		image = tf.image.resize_images(image, size=(227, 227))
		image = tf.expand_dims(image, 0)
		#image = tf.get_variable('image', shape=[1, 227, 227, 3])
		weight_bias = load_initial_params('alexnet.npy')
		res = inference(image, weight_bias, 1.0)
		sess.run(tf.global_variables_initializer())
		val = sess.run(res)
		print('result' + str(val.shape))
		print('result' + str(tf.reduce_max(val, 1).eval()))
		print('max value index ' + str(tf.argmax(val, 1).eval()))
		#print(res.eval())
	print('end')

	
