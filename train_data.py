import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2
from tqdm import tqdm
import keras

tt_data_path = r'test_data'

pickle_in = open(os.path.join(tt_data_path, 'x.pickle'),"rb")
x = pickle.load(pickle_in)

pickle_in = open(os.path.join(tt_data_path, 'y.pickle'),"rb")
y = pickle.load(pickle_in)

pickle_in = open(os.path.join(tt_data_path, 'i.pickle'),"rb")
i = pickle.load(pickle_in)

pickle_in = open(os.path.join(tt_data_path, 'j.pickle'),"rb")
j = pickle.load(pickle_in)

x = np.array(x / 255.0)
y = np.array(y)
i = np.array(i / 255.0)
j = np.array(j)

img_size = 30

x = x.reshape(x.shape[0],img_size,img_size,3)
i = i.reshape(x.shape[0],img_size,img_size,3)

model = Sequential()

model.add(Conv2D(32, (7, 7), input_shape=(img_size, img_size, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
"""
model.add(Conv2D(100, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
"""
model.add(Flatten())  
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = "binary_crossentropy",
			  optimizer = "adam",
			  metrics = ['accuracy'])

model.fit(x, y, batch_size=32, epochs=20, validation_split=0.2)

test = model.evaluate(i, j)
print(test)

model.save('model')
