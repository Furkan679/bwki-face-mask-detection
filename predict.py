import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from numpy import array
import tensorflow as tf
import cv2
from face_recognition import face_locations, load_image_file
from PIL import Image

print('\n')
path = input("please input the path: ")

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

model = tf.keras.models.load_model('model')

#path = r'datasets\pred'
predictions = []

img_size = 30

for img in os.listdir(path):

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

if len(predictions) != 1:
	print(predictions)
else:
	print('Prediction: {}'.format('Mask' if predictions[0] == 1.0 else 'No mask'))

"""
img_array = cv2.imread(os.path.join(path), cv2.IMREAD_COLOR)  
new_array = cv2.resize(img_array, (50, 50))
new_array = np.array(new_array).reshape(-1, 50, 50, 3)

prediction = model.predict(new_array)
print(float(prediction[0]))
"""