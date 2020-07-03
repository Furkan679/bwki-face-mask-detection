#import aller nötigen Module
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
from numpy import array 
import tensorflow as tf 
import cv2 
from PIL import Image 
import math

#zwingt das System, die Grafikkarte zu benutzen
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

img_size = 30

cap = cv2.VideoCapture(0) # legt den Videostreamort fest, hier ist es die WebCam mit dem Index 0
face_clsfr = cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml') # definiert den Klassifizierer, der in dem Pfad gespeichert ist
eye_cascade = cv2.CascadeClassifier('classifier/lefteye.xml')

model = tf.keras.models.load_model('model') #lädt hiermit das zuvor trainierte Model

def getDistance(eye1, eye2):
	mid1 = ((2*eye1[2] + eye1[0])/2, (2*eye1[3] + eye1[1])/2)
	mid2 = ((2*eye2[2] + eye2[0])/2, (2*eye2[3] + eye2[1])/2)

	#print(mid1, mid2)

	distance = math.sqrt((eye1[0] - eye2[0])**2 + (eye1[1] - eye2[1])**2)

	return distance

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

	return midOthers[int(distances.index(shortest))], int(distances.index(shortest))

def groupEyes(eyes):
	new_eyes = []
	for eye in eyes:
			if len(new_eyes) == 0:
				new_eyes.append((eye, (eyes[getShortestEye(eye, eyes)[1]])))

			else:
				for left, right in new_eyes:
					if eye.tolist() == left.tolist() or eye.tolist() == right.tolist():
						break
					break

				new_eyes.append((eye, (eyes[getShortestEye(eye, eyes)[1]])))
	
	return new_eyes[::2]

def getleftmosteye(eyes):
	leftmost = 9999999
	leftmostindex = -1

	for i in range(0,2):
		if eyes[i][0] < leftmost:
			leftmost = eyes[i][0]
			leftmostindex = i

	return eyes[leftmostindex]

#definiert die Methode 'predict', die 1 für Maske und 0 für keine Maske zurückgibt
def predict(img):
	global img_size
	
	new_array = cv2.resize(img, (img_size, img_size)) # das Bild wird in die nötige Größe, das heißt in die Eingabe-Form des NN (30, 30) geresized
	new_array = array(new_array).reshape(-1, img_size, img_size, 3) # das Bild wird lediglich ins richtige Format gebracht

	return model.predict(new_array)[0][0] # hier gibt die Methode zurück, was das trainierte Model zurückgibt, wenn new_array getestet wird

def slice(image, distance, x, y, w, h):
	image = image[int(y-distance/2):int(y+w+1.5*distance), int(x-0.5*distance):int(x+w+1.5*distance)]
	return image

while(True):
	ret, image = cap.read() # read() gibt einen Tupel zurück, ret ist nicht relevant
	face_location = eye_cascade.detectMultiScale(image,1.3,5) # mit Hilfe des Klassifizierers werden alle Gesichter erkannt und wie folgt gespeichert:
	# [(Gesicht 1), (Gesicht 2), ..., (Gesicht n)]
	#  (x0 Koordinate des oberen linken Punktes des Rechtecks, das das Gesicht umrahmt; y0; Länge in x Richtung von x0 aus; Länge in y Richtung von y0 aus)

	#for (x,y,w,h) in face_location: # Für jedes Tupel aller gesichteten Gesichter wird folgendes gemacht:

	if len(face_location) >= 2:

		pairs = groupEyes(face_location)
		#print(pairs)
		new_eyes = []
		distances = []

		for (left, right) in pairs:
			distances.append(getDistance(left.tolist(), right.tolist()))

		for i in pairs:
			new_eyes.append(getleftmosteye(i))

		ctr = 0
		for eye in new_eyes:
			#print(eye)
			(x, y, w, h) = eye
			print(distances[ctr])
			new_image = slice(image, distances[ctr], x, y, w, h)
			#new_image = image[y-20:y+w+100, x-20:x+w+100] # Das Bild wird zugeschnitten, damit man das Gesicht allein untersuchen kann

			pred = predict(new_image) # Das zugeschnittene Bild wird nun getestet
			#print(pred)
			text = 'Mask' if pred == 1 else 'No Mask' # Der Text, der ausgegeben wird ist 'Mask', wenn predict(new_image) 1 zurückgibt, 0 andernfalls

			cv2.rectangle(image, (int(x-0.5*distances[ctr]), int(y-distances[ctr]/2)), (int(x+w+1.5*distances[ctr]), int(y+w+1.5*distances[ctr])), (0, 255, 0)) # jetzt wird ein Rechteck aufs Bild image gemalt in der Form:
			# cv2.rectangle(Bild, auf das das RE gemalt wird; (Koordinaten des linken oberen Punktes), (Verschiebung in x und y Richtung), (Farbe als RGB-Wert))

			cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0))
			# nun wird der text über das RE geschrieben

			ctr += 1

	cv2.imshow('frame', image) # Diese Methode erzeugt ein Fenster und Inhalt ist das aktuelle Bild, der Name des Fensters ist 'frame'

	if cv2.waitKey(1) & 0xFF == ord('q'): # 'Wenn 'q' gedrückt wird, schließe alles'
		break
