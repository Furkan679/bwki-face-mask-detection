import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from tqdm import tqdm
from PIL import Image

dir = ['train', 'test']

datadir = r'C:\Users\Furkan1\Documents\GitHub\bwki-face-mask\datasets\3\observations-master\experiements\dest_folder'
categories = ['with_mask', 'without_mask']
	
trainingData = []
testData = []

img_size = 50
"""
def createTrainingData():
	global img_size
	for category in categories:
		path = os.path.join(datadir, category)
		classNum = categories.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)	
				new_array = cv2.resize(img_array, (img_size, img_size))
				trainingData.append([new_array, classNum])
			except Exception as e:
				pass
"""

def createData():
	for i in dir:
	    for category in categories: 

	    	class_num = 0

	    	path = os.path.join(datadir,i,category) 
	    	if category == 'with_mask': class_num = 1

	    	for img in tqdm(os.listdir(path)): 
	    		try:
	    			img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  
	    			new_array = cv2.resize(img_array, (img_size, img_size))  
	    			trainingData.append([new_array, class_num])  
	    		except Exception as e:  
	    			pass

	    for category in categories: 

	    	class_num = 0

	    	path = os.path.join(datadir,i,category) 
	    	if category == 'with_mask': class_num = 1

	    	for img in tqdm(os.listdir(path)): 
	    		try:
	    			img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  
	    			new_array = cv2.resize(img_array, (img_size, img_size))  
	    			testData.append([new_array, class_num])  
	    		except Exception as e:  
	    			pass



createData()

random.shuffle(trainingData)
random.shuffle(testData)

"""
for i in range(10):
	print(trainingData[i][1])
	plt.imshow(trainingData[i][0], cmap='gray')  
	plt.show()  
"""
x = []
y = []

for features, label in trainingData:
	x.append(features)
	y.append(label)

x = np.array(x).reshape(-1, img_size, img_size, 3)

pickle_out = open("test_data/x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("test_data/y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

i = []
j = []

for features, label in trainingData:
	i.append(features)
	j.append(label)

i = np.array(i).reshape(-1, img_size, img_size, 3)

pickle_out = open("test_data/i.pickle","wb")
pickle.dump(i, pickle_out)
pickle_out.close()

pickle_out = open("test_data/j.pickle","wb")
pickle.dump(j, pickle_out)
pickle_out.close()


