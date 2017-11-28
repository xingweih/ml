#!/home/hxw/anaconda3/envs/tensorflow/bin/python 
import tensorflow as tf
import numpy as np 


import person_input
import person
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default='/tmp/xwhuang/person_train',
                    help='Directory where to write event logs and checkpoint.')
parser.add_argument('--max_steps', type=int, default=10000,
                    help='Number of batches to run.')
parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')
parser.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')

def train():
	with tf.Graph().as_default():
		global_step = tf.contrib.framework.get_or_create_global_step()
		imageBatch, labelBatch = person_input.input('all_train.tfrecords')

		'''
		class _LoggerHook(tf.train.SessionRunHook):

			def begin(self):
				self._step = -1
				self._start_time = time.time()

			def before_run(self, run_context):
				self._step += 1
				return tf.train.SessionRunArgs(loss)  # Asks for loss value.

			def after_run(self, run_context, run_values):
				if self._step % FLAGS.log_frequency == 0:
					current_time = time.time()
					duration = current_time - self._start_time
					self._start_time = current_time

				loss_value = run_values.results
				examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
				sec_per_batch = float(duration / FLAGS.log_frequency)

				format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
							  'sec/batch)')
				print (format_str % (datetime.now(), self._step, loss_value,
	   				   examples_per_sec, sec_per_batch))

		with tf.train.MonitoredTrainingSession(
			checkpoint_dir=FLAGS.train_dir,
			hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
				   tf.train.NanTensorHook(loss),
			       _LoggerHook()],
		config=tf.ConfigProto(
			log_device_placement=FLAGS.log_device_placement)) as mon_sess:
		while not mon_sess.should_stop():
			mon_sess.run(train_op)
		'''

		with tf.Session() as sess:
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			images, labels = sess.run([imageBatch, labelBatch])

			logits = person.inference(images)
			total_loss = person.loss(logits, labels)
			loss_op, train_op = person.train(total_loss, global_step)

			#print(images)
			sess.run(tf.global_variables_initializer())
			for i in range(100):
				loss, train = sess.run([loss_op, train_op])
				#accuracy = np.sum(logits.eval() == labels) / 128.0
				#print(labels)
				print(logits.eval())
				print(str(i) + ' round, loss = ' + str(loss))

			coord.request_stop()
			coord.join(threads)

	print('----------------------person_train.py end----------------------')

def main(argv=None):
	train()

if __name__ == '__main__':
	FLAGS = parser.parse_args()
	tf.app.run()
