import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import os

model = tf.keras.models.load_model('model')

path = r'datasets\eigeneDateien'
predictions = []

for img in tqdm(os.listdir(path)):  
	img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)  
	new_array = cv2.resize(img_array, (50, 50))
	new_array = np.array(new_array).reshape(-1, 50, 50, 3)

	predictions.append([float(model.predict(new_array)[0]), img])


for i, j in predictions:
	print(int(i), j)

"""
img_array = cv2.imread(os.path.join(path), cv2.IMREAD_COLOR)  
new_array = cv2.resize(img_array, (50, 50))
new_array = np.array(new_array).reshape(-1, 50, 50, 3)

prediction = model.predict(new_array)
print(float(prediction[0]))
"""