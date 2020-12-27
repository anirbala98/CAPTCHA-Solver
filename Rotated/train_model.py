import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.layers import Activation
from keras.layers import LSTM
from helpers import resize_to_fit


input_folder = "Upright images"
model_filename = "captcha_model.hdf5"
model_labels_filename = "model_labels.dat"

lstm_output_size = 70

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


#X_train = X_train.reshape(20)

with open(model_labels_filename, "wb") as f:
    pickle.dump(lb, f)


    
model = Sequential()
model.add(Conv2D(8, (2, 2), padding="same", input_shape=(20,20,1), activation="relu"))
model.add(MaxPooling2D(pool_size=(1, 1), strides=(3, 3)))
model.add(Conv2D(8, (2, 2), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(1, 1), strides=(3, 3)))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(32, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=20, epochs=2, verbose=1)
model.save(model_filename)
