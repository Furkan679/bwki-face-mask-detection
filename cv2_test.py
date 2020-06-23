import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from numpy import array 
import tensorflow as tf 
import cv2 
import face_recognition 
from PIL import Image 

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

img_size = 30

cap = cv2.VideoCapture(0) 
eye_cascade = cv2.CascadeClassifier('classifier/haarcascade_eye.xml') 

model = tf.keras.models.load_model('model') 


def predict(img):
	global img_size
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	new_img = Image.fromarray(img) 
	new_img.save('pred/test.jpg') 

	img_array = cv2.imread('pred/test.jpg', cv2.IMREAD_COLOR)  
	new_array = cv2.resize(img_array, (img_size, img_size))
	new_array = array(new_array).reshape(-1, img_size, img_size, 3) 

	return model.predict(new_array)[0][0] 

while(True):
	ret, image = cap.read() 
	eyes = eye_cascade.detectMultiScale(image, 1.3, 5) 

	for (x,y,w,h) in eyes: 

		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow('frame', image) 

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
