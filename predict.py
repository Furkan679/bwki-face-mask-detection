import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import os
import cv2
import pickle
from tqdm import tqdm

model = tf.keras.models.load_model('model')


path = r'datasets\1\images\8b74444760f670bb5cee48f86bb7be950bf113a8.jpg'

img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  
new_array = cv2.resize(img_array, (50, 50))
new_array = np.array(new_array).reshape(-1, 50, 50, 1)

prediction = model.predict(new_array)
print(prediction[0])
