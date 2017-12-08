#!/home/hxw/anaconda3/envs/tensorflow/bin/python 
from xml.etree import ElementTree as ET
path = '../../../VOCdevkit/VOC2012/'
d = {
	"aeroplane": 0,
	"bicycle": 1,
	"bird": 2,
	"boat": 3,
	"bottle": 4,
	"bus": 5,
	"car": 6,
	"cat": 7,
	"chair": 8,
	"cow": 9,
	"diningtable": 10,
	"dog": 11,
	"horse": 12,
	"motorbike": 13,
	"person": 14,
	"pottedplant": 15,
	"sheep": 16,
	"sofa": 17,
	"train": 18,
	"tvmonitor": 19}
valFile = path + 'ImageSets/Main/val.txt'
fileListWrite = open(path + 'ImageSets/Main/all_val.txt', 'a+')
valFileReader = open(valFile)
line = valFileReader.readline().strip('\n')
while(line):
	f = path + 'Annotations/' + line + '.xml'
	tree = ET.ElementTree(file=f)
	root = tree.getroot()
	tmp = []
	for x in root.iter(tag='name'):
		name = x.text
		#print(f)
		#print(name)
		if(name in d and (not name in tmp)):
			index = d[name]
			fileListWrite.writelines(line + ' ' + str(index) + '\n')
			tmp.append(name)
	line = valFileReader.readline().strip('\n')
fileListWrite.close()
valFileReader.close()

'''


'''
