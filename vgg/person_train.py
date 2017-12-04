#!/home/hxw/anaconda3/envs/tensorflow/bin/python 
import tensorflow as tf
import numpy as np 
from PIL import Image
import argparse


import person_input
import person

parser = person.parser
parser.add_argument('--train_dir', type=str, default='/tmp/xwhuang/person_train',
                    help='Directory where to write event logs and checkpoint.')
parser.add_argument('--max_steps', type=int, default=10000,
                    help='Number of batches to run.')
parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')
parser.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')

NUM_CLASSES = person_input.NUM_CLASSES
batch = person_input.batch

def train():
	TRAIN_ROUND = FLAGS.train_round 
	with tf.Graph().as_default():
		global_step = tf.contrib.framework.get_or_create_global_step()
		imageBatch, labelBatch = person_input.input('all_train.tfrecords')
		thresh = tf.constant(0.5, shape=[batch, NUM_CLASSES])

		with tf.Session() as sess:
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			#tensorboard
			#test = tf.constant([1, 2, 3, 4, 10])
			#tf.summary.scalar('test', test)
			merged = tf.summary.merge_all()
			train_writer = tf.summary.FileWriter('tensorboard/', sess.graph)

			images, labels = sess.run([imageBatch, labelBatch])
			logits = person.inference(images)
			total_loss = person.loss(logits, labels)
			loss_op, train_op, lr_op = person.train(total_loss, global_step)

			#print(images)
			#print(tf.trainable_variables())
			saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
			sess.run(tf.global_variables_initializer())
			if(FLAGS.test == 0):
				if(FLAGS.continue_train == 1):
					model_file = tf.train.latest_checkpoint('model/')
					saver.restore(sess, model_file)
					print('restore model success')
				for i in range(TRAIN_ROUND):
					loss, train, lr = sess.run([loss_op, train_op, lr_op])
					#accuracy = np.sum(logits.eval() == labels) / 128.0
					#print('labels' + str(labels))
					#print('logits' + str(logits.eval()))
					print(str(i) + ' round, loss = ' + str(loss))
					print(str(i) + ' round, lr = '),
					print('%.6f' % lr)
					if(i == TRAIN_ROUND-1):
						#print('labels' + str(labels))
						resSigmoid = tf.sigmoid(logits.eval())
						np.savetxt('sigmoid.txt', sess.run(resSigmoid), fmt='%.4f')
						resCompare = tf.greater(resSigmoid, thresh)
						np.savetxt('pred.txt', sess.run(resCompare), fmt='%.4f')
						resEqual = tf.equal(resCompare, labels)
						resEqual = tf.cast(resEqual, tf.int32)
						np.savetxt('compare.txt', sess.run(resEqual), fmt='%d')
					#tensorboard
					summary = sess.run(merged)
					train_writer.add_summary(summary, i)
				saver.save(sess, 'model/train.ckpt')
				coord.request_stop()
				coord.join(threads)
				train_writer.close()
			else:#test
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
				loss, train, lr = sess.run([loss_op, train_op, lr_op])
				print('test loss = ' + str(loss))
				print('test lr = ' + str(lr))
				resSigmoid = tf.sigmoid(logits.eval())
				np.savetxt('sigmoid.txt', sess.run(resSigmoid), fmt='%.4f')
				resCompare = tf.greater(resSigmoid, thresh)
				np.savetxt('pred.txt', sess.run(resCompare), fmt='%d')
				np.savetxt('label.txt', labels, fmt='%d')
				resEqual = tf.equal(resCompare, labels)
				resEqual = tf.cast(resEqual, tf.int32)
				np.savetxt('compare.txt', sess.run(resEqual), fmt='%d')
			coord.request_stop()
			coord.join(threads)

	print('----------------------person_train.py end----------------------')

def main(argv=None):
	train()

if __name__ == '__main__':
	FLAGS = parser.parse_args()
	tf.app.run()
