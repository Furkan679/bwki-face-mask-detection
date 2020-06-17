import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import cv2
import face_recognition
import os
from PIL import Image
import math
from tqdm import tqdm
import scandir


def slice(image, face_location):
    # (y, x)
    #height = len(image)
    #length = len(image[0])

    #x = abs(face_location[3] - face_location[1])
    image = image[face_location[0] : face_location[2], face_location[3] : face_location[1]]

    height = len(image)
    length = len(image[0])

    return image

def detAndSaveImg(path, end_name, img_name):
    try:
        image = face_recognition.load_image_file(os.path.join(path, img_name)) #array
        face_location = face_recognition.face_locations(image)#.tolist() #bounds
        face_location = face_location[0]
        face_location = list(face_location)
        new_img_array = slice(image, face_location)

        new_img = Image.fromarray(new_img_array)
        new_img.save(os.path.join(path, end_name))

    except Exception as e:
        print(1)

ctr = 0

datdir = r'C:\Users\Furkan1\Documents\GitHub\bwki-face-mask\datasets\eigeneDateien'
enddir = r'C:\Users\Furkan1\Documents\GitHub\bwki-face-mask\datasets\eigeneDateienKopie'
#img_name = '17.jpg'
#detAndSaveImg(datdir, img_name, img_name)


for img in tqdm(os.listdir(datdir)):
    detAndSaveImg(datdir, os.path.join(enddir, '{}.jpg'.format(ctr)), img)
    ctr += 1
