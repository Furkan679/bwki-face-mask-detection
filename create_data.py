import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from tqdm import tqdm

datadir = r'C:\Users\Furkan1\Documents\GitHub\bwki-face-mask\datasets\3\observations-master\experiements\data'
categories = ['with_mask', 'without_mask']
	
trainingData = []

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
def createTrainingData():
    for category in categories:  # do dogs and cats

        path = os.path.join(datadir,category)  # create path to dogs and cats
        class_num = categories.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (img_size, img_size))  # resize to normalize data size
                trainingData.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))


createTrainingData()
print(len(trainingData))

random.shuffle(trainingData)

x = []
y = []

for features, label in trainingData:
	x.append(features)
	y.append(label)

x = np.array(x).reshape(-1, img_size, img_size, 1)

pickle_out = open("test_data/x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("test_data/y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


