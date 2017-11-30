#!/home/hxw/anaconda3/envs/tensorflow/bin/python

import tensorflow as tf
import numpy as np 
import person_input
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of images to process in a batch.')
FLAGS = parser.parse_args()
FLAGS.batch_size = person_input.batch

NUM_CLASSES = 20
MOVING_AVERAGE_DECAY = 0.9# The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.
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

def inference(images, weight_bias=None, dropoutRate=1.0):
	print('----------------train.inference---------------------')
	with tf.variable_scope('conv1') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64],
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(images, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv2') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 64, 64],
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32),
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
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv4') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 128, 128],
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32),
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
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(pool2, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv6') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(conv5, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv6 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv7') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32),
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
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(pool3, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv8 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv9') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(conv8, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv9 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv10') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32),
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
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(pool4, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv11 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv12') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(conv11, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv12 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv13') as scope:
		if(weight_bias == None):
			weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
				dtype=tf.float32, stddev=1e-2), name='weights')
			biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32),
				name='biases')
		conv = tf.nn.conv2d(conv12, weights, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv13 = tf.nn.relu(pre_activation, name=scope.name)

	pool5 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], 
						   strides=[1, 2, 2, 1], 
						   padding='VALID', name='pool5')

	flatten = tf.reshape(pool5, [-1, 7*7*512])
	fc6 = fc(flatten, 7*7*512, 4096, name='fc6', weight_bias=weight_bias)
		
	dropout6 = dropout(fc6, keep_prob=0.5)

	fc7 = fc(fc6, 4096, 4096, name='fc7', weight_bias=weight_bias)
	dropout7 = dropout(fc7, keep_prob=0.5)

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

def accuracy(logits, labels):
	return 100.0 - (
		100 * 
		np.sum(logits == labels) / labels.shape[0])

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
	
