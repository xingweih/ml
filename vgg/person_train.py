#!/home/hxw/anaconda3/envs/tensorflow/bin/python 
import tensorflow as tf
import numpy as np 
from PIL import Image


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

TRAIN_ROUND = 20
NUM_CLASSES = person_input.NUM_CLASSES
batch = person_input.batch

def train():
	with tf.Graph().as_default():
		global_step = tf.contrib.framework.get_or_create_global_step()
		imageBatch, labelBatch = person_input.input('all_train.tfrecords')
		thresh = tf.constant(0.5, shape=[batch, NUM_CLASSES])
		val = tf.get_variable('val', shape=[3, 3, 64, 64],
				initializer=tf.truncated_normal_initializer(stddev=0.2),
				trainable=True)
		saver = tf.train.Saver({'weights': tf.get_variable(name='conv1/weights', shape=[3, 3, 3, 64])})

		with tf.Session() as sess:
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			images, labels = sess.run([imageBatch, labelBatch])
			logits = person.inference(images)
			total_loss = person.loss(logits, labels)
			loss_op, train_op, lr_op = person.train(total_loss, global_step)

			#print(images)
			#print(tf.trainable_variables())
			sess.run(tf.global_variables_initializer())
			if(FLAGS.test == 0):
				for i in range(TRAIN_ROUND):
					loss, train, lr = sess.run([loss_op, train_op, lr_op])
					#accuracy = np.sum(logits.eval() == labels) / 128.0
					#print('labels' + str(labels))
					#print('logits' + str(logits.eval()))
					print(str(i) + ' round, loss = ' + str(loss))
					print(str(i) + ' round, lr = ' + str(lr))
					if(i == TRAIN_ROUND-1):
						print('labels' + str(labels))
						resSigmoid = tf.sigmoid(logits.eval())
						resCompare = tf.greater(resSigmoid, thresh)
						np.savetxt('pred.txt', sess.run(resCompare), fmt='%.4f')
						resEqual = tf.equal(resCompare, labels)
						resEqual = tf.cast(resEqual, tf.int32)
						np.savetxt('compare.txt', sess.run(resEqual), fmt='%d')
				coord.request_stop()
				coord.join(threads)
				saver.save(sess, 'model/train.ckpt')
			else:#test
				#model_file = tf.train.latest_checkpoint('model/')
				#saver.restore(sess, model_file)
				ckpt = tf.train.get_checkpoint_state('model/')
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)
					print('restore model success')
				else:
					print('restore model fail')
				#saver.restore(sess, 'model/train.ckpt')
				print(tf.trainable_variables())
				print(sess.run('conv1/weights:0'))

				loss, train, lr = sess.run([loss_op, train_op, lr_op])
				print('test loss = ' + str(loss))
				print('test lr = ' + str(lr))
				print('labels' + str(labels))
				resSigmoid = tf.sigmoid(logits.eval())
				resCompare = tf.greater(resSigmoid, thresh)
				np.savetxt('pred.txt', sess.run(resCompare), fmt='%.4f')
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
