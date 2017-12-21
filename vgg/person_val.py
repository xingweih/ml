#!/home/hxw/anaconda3/envs/tensorflow/bin/python 
import tensorflow as tf
import numpy as np 
from PIL import Image
import argparse

import person_input
import person_inference

parser = person_inference.parser
NUM_CLASSES = person_input.NUM_CLASSES
batch = person_input.batch

def train():
	with tf.Graph().as_default():
		globalStep = tf.contrib.framework.get_or_create_global_step()
		images, labels, indexes = person_input.input('all_val.tfrecords', False)
		thresh = tf.constant(0.5, shape=[batch, NUM_CLASSES])

		with tf.Session() as sess:
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			logits = person_inference.inference_vgg(images, dropoutRate=1.0)
			totalLoss = person_inference.loss(logits, labels)
			lossOp, trainOp, lrOp = person_inference.train(totalLoss, globalStep)
			pred = person_inference.predict(logits)
			accuracy = person_inference.accuracy(logits, labels)

			saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
			sess.run(tf.global_variables_initializer())

			#way 1
			model_file = tf.train.latest_checkpoint('model/')
			saver.restore(sess, model_file)
			print('restore model success')
			'''
			#way 2
			ckpt = tf.train.get_checkpoint_state('model/')
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print('restore model success')
			else:
				print('restore model fail')
			'''
			'''
			#way 3
			saver.restore(sess, 'model/train.ckpt')
			'''
			listIn = [logits, pred, labels, indexes, accuracy]
			#loss, train, lr = sess.run([loss_op, train_op, lr_op])
			for i in range(10):
				listOut = sess.run(listIn)
				#list_out = [loss, train, lr, logits_out, labels_out]
				print('image index = ' + str(listOut[3]))
				print('accuracy = ' + str(listOut[4]))
			np.savetxt('logits.txt', listOut[0], fmt='%.2f')
			np.savetxt('pred.txt', listOut[1], fmt='%d')
			np.savetxt('labels.txt', listOut[2], fmt='%d')
			coord.request_stop()
			coord.join(threads)

	print('----------------------person_train.py end----------------------')

def main(argv=None):
	train()

if __name__ == '__main__':
	FLAGS = parser.parse_args()
	tf.app.run()
