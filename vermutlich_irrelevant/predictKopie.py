import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from numpy import array
import tensorflow as tf
import cv2
import face_recognition 
from PIL import Image

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

cap = cv2.VideoCapture(0)
face_clsfr=cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')

model = tf.keras.models.load_model('model')

img_size = 30
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

def predict(path , img_size, model, image = None, face_loc = None, video = False):

	if not video:
		predictions = []
		for img in os.listdir(path):
			try:
				image = load_image_file(os.path.join(path, img)) 

				face_location = face_locations(image)
				face_location = list(face_location[0])

				image = image[face_location[0] : face_location[2], face_location[3] : face_location[1]]

				new_img = Image.fromarray(image)
				new_img.save(os.path.join(path, img))

				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)  
				new_array = cv2.resize(img_array, (img_size, img_size))
				new_array = array(new_array).reshape(-1, img_size, img_size, 3)

				predictions.append(model.predict(new_array)[0][0])
				return predictions

			except:
				pass

	else:

		#img = image[face_loc[0][0] : face_loc[0][2], face_loc[0][3] : face_loc[0][1]]
		img = image
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		new_img = Image.fromarray(img)

		new_img.save('pred/test.jpg')

		img_array = cv2.imread('pred/test.jpg', cv2.IMREAD_COLOR)  
		new_array = cv2.resize(img_array, (img_size, img_size))
		new_array = array(new_array).reshape(-1, img_size, img_size, 3)

		return model.predict(new_array)[0][0]


while(True):
	ret, image = cap.read()
	face_location=face_clsfr.detectMultiScale(image,1.3,5)  

	for (x,y,w,h) in face_location:
		top_left = (face_location[0][3], face_location[0][0])
		bottom_right = (face_location[0][1], face_location[0][2])
		new_image = image[y:y+w,x:x+w]  
		pred = predict('', img_size, model, new_image, face_location, video = True)
		text = 'Mask' if pred == 1.0 else 'No mask'
		cv2.rectangle(image,(x,y),(x+w,y+h), (0, 255, 0))
		cv2.putText(image, text, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0))

	cv2.imshow('frame', image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



#path = r'datasets\pred'
#predictions = []

#pred = predict(path, img_size, model)
"""
if pred != None:
	if len(pred) != 1:
		print(pred)
	else:
		print('Prediction: {}'.format('Mask' if pred[0] == 1.0 else 'No mask'))

input("Press any key")
"""
