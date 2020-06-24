#import aller nötigen Module
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
from numpy import array 
import tensorflow as tf 
import cv2 
#import face_recognition 
from PIL import Image 
import math
import random 

#zwingt das System, die Grafikkarte zu benutzen
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

img_size = 30

path = r'pred\1.jpg'

eye_cascade = cv2.CascadeClassifier('classifier/lefteye.xml')

model = tf.keras.models.load_model('model') #lädt hiermit das zuvor trainierte Model

def getShortestEye(eye, eyes):
	shortest = 2147483647
	midCurrent = (eye[0], eye[1])

	midOthers = []

	eyes = eyes.tolist()

	for eye in eyes:
		midOthers.append((eye[0], eye[1])) 

	distances = []

	for distance in midOthers:
		currentDistance = math.sqrt((distance[0] - midCurrent[0])**2 + (distance[1] - midCurrent[1])**2)
		distances.append(currentDistance)
		if currentDistance < shortest and currentDistance != 0:
			shortest = currentDistance
	#print(shortest)
	return midOthers[int(distances.index(shortest))], int(distances.index(shortest)), shortest

def groupEyes(eyes):
	new_eyes = []
	for eye in eyes:
			if len(new_eyes) == 0:
				new_eyes.append((eye, eyes[getShortestEye(eye, eyes)[1]]))

			else:
				for left, right in new_eyes:
					if eye.tolist() == left.tolist() or eye.tolist() == right.tolist():
						break
					break

				new_eyes.append((eye, eyes[getShortestEye(eye, eyes)[1]]))
	#print(len(new_eyes))	
	#print(new_eyes[::2])	
	return new_eyes[::2], new_eyes

def getleftmosteye(eyes):
	leftmost = 9999999
	leftmostindex = -1

	for i in range(0,2):
		if eyes[i][0] < leftmost:
			leftmost = eyes[i][0]
			leftmostindex = i

	return eyes[leftmostindex]

def slice(image, eye_distance, x, y, w, h):
	image = image[y-int(eye_distance/2):y+w+int(eye_distance), x-int(eye_distance/2):x+w+int(3*eye_distance/2)]

	return image

#definiert die Methode 'predict', die 1 für Maske und 0 für keine Maske zurückgibt
def predict(img):
	global img_size
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # konvertiert das Farbspektrum von BGR zu RGB, weil das Dataset in RGB ist, cv2 aber mit BGR arbeitet

	new_img = Image.fromarray(img) # Konvertiert ein Numpy-array zu einem Bild
	randomNumber = random.randint(0, 25000)
	new_img.save('pred/{}.jpg'.format(randomNumber)) # und speichert dieses als jpg ab

	img_array = cv2.imread('pred/{}.jpg'.format(randomNumber), cv2.IMREAD_COLOR)  # um es hier einzulesen
	new_array = cv2.resize(img_array, (img_size, img_size)) # das Bild wird in die nötige Größe, das heißt in die Eingabe-Form des NN (30, 30) geresized
	new_array = array(new_array).reshape(-1, img_size, img_size, 3) # das Bild wird lediglich ins richtige Format gebracht

	return model.predict(new_array)[0][0] # hier gibt die Methode zurück, was das trainierte Model zurückgibt, wenn new_array getestet wird
k = True
while k == True:
	print(1)
	#ret, image = cap.read() # read() gibt einen Tupel zurück, ret ist nicht relevant
	image = cv2.imread(path, cv2.IMREAD_COLOR)
	face_location = eye_cascade.detectMultiScale(image,1.3,5) # mit Hilfe des Klassifizierers werden alle Gesichter erkannt und wie folgt gespeichert:
	# [(Gesicht 1), (Gesicht 2), ..., (Gesicht n)]
	#  (x0 Koordinate des oberen linken Punktes des Rechtecks, das das Gesicht umrahmt; y0; Länge in x Richtung von x0 aus; Länge in y Richtung von y0 aus)

	#for (x,y,w,h) in face_location: # Für jedes Tupel aller gesichteten Gesichter wird folgendes gemacht:

	if len(face_location) >= 2:
		distances = groupEyes(face_location)[2]
		pairs = groupEyes(face_location)[0]
		new_eyes = []
		#print(distances)
		#print(len(distances))
		for i in pairs:
			new_eyes.append(getleftmosteye(i))

		#print(new_eyes)
		ctr = 0
		for eye in new_eyes:
			(x, y, w, h) = eye
			#new_image = image[y-20:y+w+100, x-20:x+w+100] # Das Bild wird zugeschnitten, damit man das Gesicht allein untersuchen kann
			slice(image, distances[ctr], x, y, w, h)
			pred = predict(new_image) # Das zugeschnittene Bild wird nun getestet
			#print(pred)
			text = 'Mask' if pred == 1 else 'No Mask' # Der Text, der ausgegeben wird ist 'Mask', wenn predict(new_image) 1 zurückgibt, 0 andernfalls

			cv2.rectangle(image, (x-20, y-20), (x+w+100, y+h+100), (0, 255, 0)) # jetzt wird ein Rechteck aufs Bild image gemalt in der Form:
			# cv2.rectangle(Bild, auf das das RE gemalt wird; (Koordinaten des linken oberen Punktes), (Verschiebung in x und y Richtung), (Farbe als RGB-Wert))

			cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0))
			# nun wird der text über das RE geschrieben

			new_img = Image.fromarray(image) # Konvertiert ein Numpy-array zu einem Bild

			new_img.save('pred/test.jpg')
			ctr+=1
		k = False


