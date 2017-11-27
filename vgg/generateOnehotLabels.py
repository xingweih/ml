#!/home/hxw/anaconda3/envs/tensorflow/bin/python 
import tensorflow as tf
import numpy as np 
from PIL import Image
import matplotlib as plt

IMAGES_PER_CLASS = 5717
NUM_CLASSES = 20

def main():
	lineNum = 0
	path = '../../../VOCdevkit/VOC2012/'
	fileList = open(path + 'ImageSets/Main/all_train_spase_labels.txt')
	fileListWrite = open(path + 'ImageSets/Main/all_train_onehot_labels.txt', 'w')
	line = fileList.readline()
	d = {}
	with tf.Session() as sess:
		while(line):
			imageName = line.split()[0]
			label = line.split()[1]
			label = int(label)
			tmp = tf.one_hot(label, NUM_CLASSES, 1, 0).eval()
			if(imageName in d.keys()):
				d[imageName] = tmp + d[imageName]
			else:
				d[imageName] = tmp

			'''
			fileListWrite.writelines(imageName + ' ' +
									 str(d[imageName]) + '\n')
			'''
			line = fileList.readline()
			lineNum += 1
			if(lineNum % 100 == 0):
				print(lineNum)
		fileList.close()
		for key in d:
			fileListWrite.writelines(key + ' ' +
									 str(d[key]) + '\n')
		fileListWrite.close()


if __name__ == '__main__':
	print('---------------------------BEGIN-------------------------')
	main()
	print('----------------------------END--------------------------')

