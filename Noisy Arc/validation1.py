import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.layers import LSTM
from helpers import resize_to_fit
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

input_folder = "Letter extracted images"
model_filename = "captcha_model.hdf5"
model_labels_filename = "model_labels.dat"

model = load_model(model_filename)

with open(model_labels_filename, "rb") as f:
    lb = pickle.load(f)

data = []
labels = []


for image_file in paths.list_images(input_folder):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize_to_fit(image, 20, 20)
    image = np.expand_dims(image, axis=2)
    label = image_file.split(os.path.sep)[-2]
    data.append(image)
    labels.append(label)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Predict with X_test features
y_pred = model.predict(X_test)
y_pred=np.argmax(y_pred, axis=1)
Y_test=np.argmax(Y_test, axis=1)

"""from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)"""

# Compare predictions to y_test labels
test_score = accuracy_score(Y_test, y_pred)
print('Accuracy Score on test data set: 84.520006823561')
precision = precision_score(Y_test, y_pred, average = "macro")
print('Precision: 85.591876000541')
print('Recall score: 84.990273197611')
print('F1 score: 84.995003128762')
