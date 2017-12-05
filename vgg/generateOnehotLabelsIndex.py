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
	fileList = open(path + 'ImageSets/Main/all_train_onehot_labels.txt')
	fileListWrite = open(path + 'ImageSets/Main/all_train_onehot_labels_index.txt', 'w')
	line = fileList.readline()
	lineNum = 0
	while(line):
		newLine = line + ' ' + str(lineNum) + '\n'
		fileListWrite.writelines(newLine)
		line = fileList.readline()
		lineNum += 1
	fileList.close()
	fileListWrite.close()


if __name__ == '__main__':
	print('---------------------------BEGIN-------------------------')
	main()
	print('----------------------------END--------------------------')

