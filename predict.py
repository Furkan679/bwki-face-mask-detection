import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2

model = tf.keras.models.load_model('model')

path = r'datasets\3\observations-master\experiements\dest_folder\train\without_mask\1.jpg'

img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  
new_array = cv2.resize(img_array, (50, 50))
new_array = np.array(new_array).reshape(-1, 50, 50, 1)

prediction = model.predict(new_array)
print(prediction)
