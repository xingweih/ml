#!/home/hxw/anaconda3/envs/tensorflow/bin/python 
import tensorflow as tf
import numpy as np 
from PIL import Image
import matplotlib as plt

IMAGES_PER_CLASS = 5717
NUM_CLASSES = 20

def train():
	lineNum = 0
	path = '../../../VOCdevkit/VOC2012/'
	fileList = open(path + 'ImageSets/Main/all_train_onehot_labels.txt')
	fileListWrite = open(path + 'ImageSets/Main/all_train_onehot_labels_index.txt', 'w')
	line = fileList.readline().strip('\n')
	lineNum = 0
	while(line):
		newLine = line + ' ' + str(lineNum) + '\n'
		fileListWrite.writelines(newLine)
		line = fileList.readline().strip('\n')
		lineNum += 1
	fileList.close()
	fileListWrite.close()

def val():
	lineNum = 0
	path = '../../../VOCdevkit/VOC2012/'
	fileList = open(path + 'ImageSets/Main/all_val_onehot_labels.txt')
	fileListWrite = open(path + 'ImageSets/Main/all_val_onehot_labels_index.txt', 'w')
	line = fileList.readline().strip('\n')
	lineNum = 0
	while(line):
		newLine = line + ' ' + str(lineNum) + '\n'
		fileListWrite.writelines(newLine)
		line = fileList.readline().strip('\n')
		lineNum += 1
	fileList.close()
	fileListWrite.close()

def main():
	#train()
	val()

if __name__ == '__main__':
	print('---------------------------BEGIN-------------------------')
	main()
	print('----------------------------END--------------------------')

