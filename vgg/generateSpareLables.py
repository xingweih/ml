#!/home/hxw/anaconda3/envs/tensorflow/bin/python 
import tensorflow as tf
import numpy as np 
from PIL import Image
import matplotlib as plt

IMAGES_PER_CLASS = 5717

def main():
	lineNum = 0
	path = '../../../VOCdevkit/VOC2012/'
	fileList = open(path + 'ImageSets/Main/all_train.txt')
	fileListWrite = open(path + 'ImageSets/Main/all_train_spase_labels.txt', 'w')
	line = fileList.readline()
	while(line):
		imageName = line.split()[0]
		label = line.split()[1]
		label = int(label)
		if(label == 1):
			sparseLabel = int(lineNum / IMAGES_PER_CLASS)
			fileListWrite.writelines(imageName + ' ' +
									 str(sparseLabel) + '\n')
		line = fileList.readline()
		lineNum += 1
	fileList.close()
	fileListWrite.close()

if __name__ == '__main__':
	print('---------------------------BEGIN-------------------------')
	main()
	print('----------------------------END--------------------------')

