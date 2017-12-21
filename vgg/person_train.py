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
height = person_input.height
width = person_input.width
channel = person_input.channel

def train():
	TRAIN_ROUND = FLAGS.train_round 
	with tf.Graph().as_default():
		globalStep = tf.contrib.framework.get_or_create_global_step()
		thresh = tf.constant(0.5, shape=[batch, NUM_CLASSES])

		imagesTrain, labelsTrain, indexesTrain = person_input.input('all_train.tfrecords', True, True)
		imagesVal, labelsVal, indexesVal = person_input.input('all_val.tfrecords', False, False)

		imagesIsTrain = tf.placeholder(dtype=bool, shape=())
		labelsIsTrain = tf.placeholder(dtype=bool, shape=())
		indexesIsTrain = tf.placeholder(dtype=bool, shape=())
		dropoutIsTrain = tf.placeholder(dtype=bool, shape=())
		
		images = tf.cond(imagesIsTrain, lambda: imagesTrain, lambda: imagesVal)
		labels = tf.cond(labelsIsTrain, lambda: labelsTrain, lambda: labelsVal)
		indexes = tf.cond(indexesIsTrain, lambda: indexesTrain, lambda: indexesVal)
		dropoutRate = tf.cond(dropoutIsTrain, lambda: 0.5, lambda: 1.0)

		with tf.Session() as sess:
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			#tensorboard
			#test = tf.constant([1, 2, 3, 4, 10])
			#tf.summary.scalar('test', test)
			logits = person_inference.inference_vgg(images, dropoutRate=dropoutRate)
			totalLoss = person_inference.loss(logits, labels)
			lossOp, trainOp, lrOp = person_inference.train(totalLoss, globalStep)
			pred = person_inference.predict(logits)
			accuracy = person_inference.accuracy(logits, labels)

			merged = tf.summary.merge_all()
			train_writer = tf.summary.FileWriter('tensorboard/', sess.graph)

			#print(images)
			#print(tf.trainable_variables())
			saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
			sess.run(tf.global_variables_initializer())
			if(FLAGS.continue_train == 1):
				model_file = tf.train.latest_checkpoint('model/')
				saver.restore(sess, model_file)
				print('restore model success')
			for i in range(TRAIN_ROUND):
				listInTrain = [lossOp, logits, pred, labels, accuracy, indexes, trainOp, lrOp]
				listInVal = [totalLoss, logits, pred, labels, accuracy, indexes]
				if(i % 20 != 0):
					listOut = sess.run(listInTrain, 
								feed_dict={imagesIsTrain : True,
										   labelsIsTrain : True,
										   indexesIsTrain: True,
										   dropoutIsTrain: True})
				else:
					listOut = sess.run(listInVal, 
								feed_dict={imagesIsTrain : False,
										   labelsIsTrain : False,
										   indexesIsTrain: False,
										   dropoutIsTrain: False})
				#print('labels' + str(labels.eval()))
				#print('index ' + str(indexes.eval()))
				#print('logits' + str(logits.eval()))
				#print(str(i) + ' round, lr = ', end='')
				#print('%.6f' % listOut[2])
				#print(str(i) + ' round, Image index = ' + str(listOut[5]))
				if(i % 20 != 0):
					print(str(i) + ' round, Train loss = ' + str(listOut[0]))
					print(str(i) + ' round, Train accuracy = ' + str(listOut[4]))
				else:
					print(str(i) + ' round, Eval loss = ' + str(listOut[0]))
					print(str(i) + ' round, Eval accuracy = ' + str(listOut[4]))

				#tensorboard
				#summary = sess.run(merged)
				#train_writer.add_summary(summary, i)
			saver.save(sess, 'model/train.ckpt')
			coord.request_stop()
			coord.join(threads)
			train_writer.close()
			coord.request_stop()
			coord.join(threads)

	print('----------------------person_train.py end----------------------')

def main(argv=None):
	train()

if __name__ == '__main__':
	FLAGS = parser.parse_args()
	tf.app.run()
