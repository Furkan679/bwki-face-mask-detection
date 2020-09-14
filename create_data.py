import random
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm

DATADIR = 'dataset'
CATEGORIES = ['with_mask', 'without_mask']
IMG_SIZE = 30

trainingData = []

def createData():
	w, wo = 0, 0
	for category in CATEGORIES: 
		try:
			class_num = 0

			path = DATADIR + '/' + category
			if category == 'with_mask': class_num = 1

			for img in tqdm(os.listdir(path)): 
				try:
					img_array = cv2.imread(path + '/' + img ,cv2.IMREAD_COLOR)  
					new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
					trainingData.append([new_array, class_num])  
					if class_num == 1:
						w+=1
					else:
						wo+=1
				except Exception as e:  
					print(1)
					pass
		except:
			pass
		pass
	print(f"With mask: {w}\nWithout mask: {wo}")

createData()

random.shuffle(trainingData)

x, y = [], []

for features, label in trainingData:
	x.append(features)
	y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

np.save("train_and_test_data/x.npy", x)
np.save("train_and_test_data/y.npy", y)

