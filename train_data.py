import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import cv2
from tqdm import tqdm
import keras

IMG_SIZE = 30

x = np.load("train_and_test_data/x.npy")
y = np.load("train_and_test_data/y.npy")

x = np.array(x / 255.0)
y = np.array(y)



x = x.reshape(x.shape[0], IMG_SIZE, IMG_SIZE, 3)

model = Sequential([Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation = 'tanh'),
					MaxPooling2D(pool_size=(2, 2)),
					Flatten(),
					Dense(256, activation = 'relu'),
					Dense(1, activation = 'sigmoid')])

model.compile(loss = "binary_crossentropy",
			  optimizer = "adam",
			  metrics = ['accuracy'])

model.fit(x, y, batch_size=32, epochs=20, validation_split=0.2)
model.save('model')
