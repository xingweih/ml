#!/home/hxw/anaconda3/envs/tensorflow/bin/python 
import tensorflow as tf
import numpy as np 
from PIL import Image
import matplotlib as plt

width = 224
height = 224

def createTFRecords():
	writer = tf.python_io.TFRecordWriter("person_train.tfrecords")
	path = '../../../VOCdevkit/VOC2012/'
	fileList = open(path + 'ImageSets/Main/person_train.txt')
	line = fileList.readline()
	while(line):
		imageName = line.split()[0]
		label = line.split()[1]
		label = int(label)
		label = int(float(label) / 2 + 0.5)
		#print(label)
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
		'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
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

