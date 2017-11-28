#!/home/hxw/anaconda3/envs/tensorflow/bin/python 
import tensorflow as tf
import numpy as np 
from PIL import Image
import matplotlib as plt

width = 224
height = 224
NUM_CLASSES = 20

def createTFRecords():
	writer = tf.python_io.TFRecordWriter("all_train.tfrecords")
	path = '../../../VOCdevkit/VOC2012/'
	fileList = open(path + 'ImageSets/Main/all_train_onehot_labels.txt')
	line = fileList.readline()
	while(line):
		imageName = line.split()[0]
		label = []
		for i in range(NUM_CLASSES):
			label.append(int(line.split()[i+1]))
		labelBytes = bytes(label)
		imageName = path + 'JPEGImages/' + imageName + '.jpg'
		image = Image.open(imageName)
		image = image.resize((width, height))
		imageBytes = image.tobytes()
		'''
		imageRaw = tf.gfile.FastGFile(imageName, 'rb').read()
		image = tf.image.decode_jpeg(imageRaw)
		image = tf.image.resize_images(image, size=[width, height])
		'''
		example = tf.train.Example(features=tf.train.Features(feature={
		'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labelBytes])),
		'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imageBytes]))
		}))
		writer.write(example.SerializeToString())
		line = fileList.readline()
	fileList.close()
	writer.close()

def main():
	with tf.Session() as sess:
		createTFRecords()

if __name__ == '__main__':
	print('---------------------------BEGIN-------------------------')
	main()
	print('----------------------------END--------------------------')

