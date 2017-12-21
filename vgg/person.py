#!/home/hxw/anaconda3/envs/tensorflow/bin/python

import tensorflow as tf
import numpy as np 
import person_input
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of images to process in a batch.')
parser.add_argument('--train_round', type=int, default=100,
                    help='How many round to train')
parser.add_argument('--continue_train', type=int, default=1,
                    help='whether to use last result to continue to train')
FLAGS = parser.parse_args()
FLAGS.batch_size = person_input.batch

NUM_CLASSES = 20
MOVING_AVERAGE_DECAY = 0.9# The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.96  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-2	# Initial learning rate.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000

def fc(x, sizeIn, sizeOut, name, relu=True, weight_bias=None):
	with tf.variable_scope(name) as scope:
		if(weight_bias == None):
			weights = tf.get_variable(name='weights', shape=[sizeIn, sizeOut],
									  initializer=tf.contrib.layers.xavier_initializer(),
									  trainable=True)
			biases = tf.get_variable('biases', shape=[sizeOut],
									 initializer=tf.constant_initializer(0.1),
									 trainable=True)
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


def generate_variables(name, shape, initializer):
	return tf.get_variable(name=name,
 						   shape=shape,
						   initializer=initializer)

def generate_weight(name, shape, stddev, wd=5e-4):
	'''
	return generate_variables(name, shape,
		initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	'''
	var = tf.get_variable(name=name, shape=shape, 
						   initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						  )
	if wd is not None:
		#print('add l2_loss')
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	
	return var

def generate_bias(name, shape, constant):
	return generate_variables(name, shape,
		initializer=tf.constant_initializer(constant))



def inference_alex(images, weight_bias=None, dropoutRate=1.0):
	with tf.variable_scope('conv1') as scope:
		if(weight_bias == None):
			kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96],
				dtype = tf.float32, stddev=1e-2), name='weights')
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
				dtype = tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32), name='biases')
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
				dtype = tf.float32, stddev=1e-2), name='weights')
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
				dtype = tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[384], dtype=tf.float32), name='biases')
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
				dtype = tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32), name='biases')
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

	fc8 = fc(fc7, 4096, NUM_CLASSES, name='fc8', relu=False, weight_bias=weight_bias)
	dropout8 = dropout(fc8, dropoutRate)
	return dropout8

def inference_vgg(images, weight_bias=None, dropoutRate=1.0):
	print('----------------train.inference---------------------')
	with tf.variable_scope('conv1') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 3, 64], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[64], constant=0.0)
		conv = tf.nn.conv2d(images, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv2') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 64, 64], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[64], constant=0.1)
		conv = tf.nn.conv2d(conv1, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)
	
	pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], 
						   strides=[1, 2, 2, 1], 
						   padding='VALID', name='pool1')

	with tf.variable_scope('conv3') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 64, 128], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[128], constant=0.1)
		conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv4') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 128, 128], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[128], constant=0.1)
		conv = tf.nn.conv2d(conv3, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(pre_activation, name=scope.name)

	pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], 
						   strides=[1, 2, 2, 1], 
						   padding='VALID', name='pool2')

	with tf.variable_scope('conv5') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 128, 256], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[256], constant=0.1)
		conv = tf.nn.conv2d(pool2, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv6') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 256, 256], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[256], constant=0.1)
		conv = tf.nn.conv2d(conv5, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv6 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv7') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 256, 256], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[256], constant=0.1)
		conv = tf.nn.conv2d(conv6, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv7 = tf.nn.relu(pre_activation, name=scope.name)

	pool3 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1], 
						   strides=[1, 2, 2, 1], 
						   padding='VALID', name='pool3')

	with tf.variable_scope('conv8') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 256, 512], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[512], constant=0.1)
		conv = tf.nn.conv2d(pool3, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv8 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv9') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 512, 512], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[512], constant=0.1)
		conv = tf.nn.conv2d(conv8, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv9 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv10') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 512, 512], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[512], constant=0.1)
		conv = tf.nn.conv2d(conv9, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv10 = tf.nn.relu(pre_activation, name=scope.name)

	pool4 = tf.nn.max_pool(conv10, ksize=[1, 2, 2, 1], 
						   strides=[1, 2, 2, 1], 
						   padding='SAME', name='pool4')

	with tf.variable_scope('conv11') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 512, 512], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[512], constant=0.1)
		conv = tf.nn.conv2d(pool4, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv11 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv12') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 512, 512], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[512], constant=0.1)
		conv = tf.nn.conv2d(conv11, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv12 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv13') as scope:
		if(weight_bias == None):
			weights = generate_weight(name='weights', shape=[3, 3, 512, 512], stddev=1e-2)
			biases = generate_bias(name='biases', shape=[512], constant=0.1)
		conv = tf.nn.conv2d(conv12, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv13 = tf.nn.relu(pre_activation, name=scope.name)

	pool5 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], 
						   strides=[1, 2, 2, 1], 
						   padding='VALID', name='pool5')

	flatten = tf.reshape(pool5, [-1, 7*7*512])
	fc6 = fc(flatten, 7*7*512, 4096, name='fc6', weight_bias=weight_bias)
		
	dropout6 = dropout(fc6, keep_prob=dropoutRate)

	fc7 = fc(fc6, 4096, 4096, name='fc7', weight_bias=weight_bias)
	dropout7 = dropout(fc7, keep_prob=dropoutRate)

	fc8 = fc(fc7, 4096, NUM_CLASSES, name='fc8', 
			 relu=False, weight_bias=weight_bias)
	dropout8 = dropout(fc8, dropoutRate)
	return dropout8

def loss(logits, labels):
	print('----------------train.loss---------------------')
	labels = tf.cast(labels, tf.float32)
	'''
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
						labels=labels, logits=logits, name='cross_entropy_per_example')
	'''
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
						labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def predict(logits):
	thresh = tf.constant(0.5, shape=[FLAGS.batch_size, NUM_CLASSES])
	resSigmoid = tf.sigmoid(logits)
	resCompare = tf.greater(resSigmoid, thresh)
	return tf.cast(resCompare, tf.float32)

def accuracy(logits, labels):
	ones = tf.ones([FLAGS.batch_size], tf.int32)
	pred = predict(logits)
	isEqual = tf.equal(pred, labels)
	isEqual = tf.cast(isEqual, tf.int32)
	mean = tf.reduce_mean(isEqual, 1)
	mean = tf.equal(mean, ones)
	return tf.cast(mean, tf.int32)
	#return tf.reduce_mean(isEqual)
	

def _add_loss_summaries(total_loss):
	print('----------------train._add_loss_summaries---------------------')
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])
	for l in losses + [total_loss]:
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))
	return loss_averages_op

def train(total_loss, global_step):
	print('----------------train.train---------------------')
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
									global_step,
									decay_steps,
									LEARNING_RATE_DECAY_FACTOR,
									staircase=True)
	tf.summary.scalar('learning_rate', lr)
	loss_averages_op = _add_loss_summaries(total_loss)

	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)
	
	apply_gradient_op = opt.apply_gradients(grads, global_step)

	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + '/gradients', grad)

	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')
	return total_loss, train_op, lr
		
	
if __name__ == '__main__':
	print('---------------Program Begin--------------------')
	imageBatch, labelBatch = person_input.input('all_train.tfrecords')
	with tf.Session() as sess:

		image_raw = tf.gfile.FastGFile('pic.jpg', 'rb').read()
		image = tf.image.decode_jpeg(image_raw, channels=3)
		image = tf.image.resize_images(image, size=(224, 224))
		image = tf.expand_dims(image, 0)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		images, labels = sess.run([imageBatch, labelBatch])

		#print(images)
		inference = inference(images)
		sess.run(tf.global_variables_initializer())
		res = sess.run(inference)
		print(res)

		coord.request_stop()
		coord.join(threads)
	print('---------------Program End--------------------')
	
