#!/home/hxw/anaconda3/envs/tensorflow/bin/python 
import tensorflow as tf
import numpy as np 
from PIL import Image
import matplotlib as plt

width = 224
height = 224
channel = 3
batch = 1 
numExaplesPerEpoch = 2000

def readOneImage(fileNameQueue):
	reader = tf.TFRecordReader()
	_, serializedExample = reader.read(fileNameQueue)
	features = tf.parse_single_example(serializedExample,
									   features={
									   	'label': tf.FixedLenFeature([], tf.int64),
									   	'image': tf.FixedLenFeature([], tf.string),
									   })
	image = tf.decode_raw(features['image'], tf.uint8)
	label = tf.cast(features['label'], tf.int32)
	return image, label

def generateBatchImageLabel(image, label, minQueueNum, batchSize, shuffle):
	numPreprocessThreashs = 16
	if(shuffle):
		images, labels = tf.train.shuffle_batch(
			[image, label],
			batch_size=batchSize,
			num_threads=numPreprocessThreashs,
			capacity=minQueueNum + 3 * batchSize,
			min_after_dequeue=minQueueNum)
	else:
		images, labels = tf.train.batch(
			[image, label],
			batch_size=batchSize,
			num_threads=numPreprocessThreashs,
			capacity=minQueueNum + 3 * batchSize)
	return images, tf.reshape(labels, [batchSize])

def input(TFRecordsFile):
	fileNameQueue = tf.train.string_input_producer([TFRecordsFile])
	image, label = readOneImage(fileNameQueue)
	
	imageFloat = tf.cast(image, tf.float32)
	labelFloat = tf.cast(label, tf.float32)
	imageFloat = tf.reshape(imageFloat, [width, height, channel])
	labelFloat = tf.reshape(labelFloat, [1])

	#imageFloat = tf.image.per_image_standardization(imageFloat)
	minFractionInQueue = 0.4
	minQueueExample = int(minFractionInQueue * numExaplesPerEpoch) 
	return generateBatchImageLabel(imageFloat, labelFloat, minQueueExample,
								   batch, shuffle=True)

def main():
	imageBatch, labelBatch = input('person_train.tfrecords')
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		sess.run(init)
		threads = tf.train.start_queue_runners(coord=coord)
		image, label = sess.run([imageBatch, labelBatch])
		print(type(image))
		print(label)
		coord.request_stop()
		coord.join(threads)
	'''
	plt.figure()
	plt.imshow(image)
	plt.show()
	'''

if __name__ == '__main__':
	print('---------------------------BEGIN-------------------------')
	main()
	print('----------------------------END--------------------------')

