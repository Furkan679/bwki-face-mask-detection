import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import os
import cv2
import pickle
from tqdm import tqdm

pickle_in = open(r"test_data/x.pickle","rb")
x = pickle.load(pickle_in)

pickle_in = open("test_data/y.pickle","rb")
y = pickle.load(pickle_in)

x = np.array(x / 255.0)
y = np.array(y)

x = x.reshape(x.shape[0],50,50,1)

print(len(x), len(y))

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=(50, 50, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss = "binary_crossentropy",
			  optimizer = "adam",
			  metrics = ['accuracy'])

model.fit(x, y, batch_size=1, epochs=4, validation_split=0.1)

model.save('model')


