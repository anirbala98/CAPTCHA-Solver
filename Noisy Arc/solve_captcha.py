from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
from PIL import Image


model_filename = "captcha_model.hdf5"
model_labels_filename = "model_labels.dat"
input_folder = "Noisy samples"

with open(model_labels_filename, "rb") as f:
    lb = pickle.load(f)

model = load_model(model_filename)

def solve(image_file):
    image = cv2.imread(image_file)
    kernel = np.ones((5,5), np.uint8)
    #cv2.imshow("Input", image)
    erosion_image = cv2.erode(image, kernel, iterations=1) 
    img = cv2.dilate(image, kernel, iterations=1)
    img = Image.fromarray(img)

    predictions = []

    width, height = img.size
    left = 0
    top = 0
    right = (width-1)/4
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))
    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    im1 = resize_to_fit(im1, 20, 20)
   
    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)
    print(prediction)
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    ###############################################################

    left = (width-1)/4
    top = 0
    right = (width-1)/6*2
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))
    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = resize_to_fit(im1, 20, 20)

    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)
    print(prediction)
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    ##################################################################

    left = (width-1)/6*2
    top = 0
    right = (width-1)/6*2.7
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))

    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = resize_to_fit(im1, 20, 20)

    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)
    print(prediction)
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    ######################################################################

    left = (width-1)/6*2.7
    top = 0
    right = (width-1)/6*3.3
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))
    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = resize_to_fit(im1, 20, 20)

    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)
    print(prediction)
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    #######################################################################

    left = (width-1)/6*3.3
    top = 0
    right = (width-1)/6*4
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))
    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = resize_to_fit(im1, 20, 20)

    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)

    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)
    print(prediction)

    captcha_text = "".join(predictions)
    return captcha_text
    cv2.waitKey()
