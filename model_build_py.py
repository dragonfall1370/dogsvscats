import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import os
import pathlib
current_path = os.getcwd()

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


X = pickle.load(open(os.path.join(current_path, "X.pickle"), "rb"))
y = pickle.load(open(os.path.join(current_path, "y.pickle"), "rb"))

X=np.array(X/255.0)
y=np.array(y)

print(X.shape[1:])


model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = "binary_crossentropy", optimizer = "adam"
              ,metrics = ['accuracy'])

model.fit(X,y, batch_size = 64,  epochs=48, validation_split = 0.3)



IMG_SIZE = 120
x_test = X[710]
x_test_pr = np.array(x_test).reshape(-1,IMG_SIZE, IMG_SIZE,1)

y_test = y[710]

x_pred= model.predict_classes(x_test_pr)

print(y_test, x_pred)

import matplotlib.pyplot as plt
import cv2


final_array = cv2.resize(x_test*255, (IMG_SIZE, IMG_SIZE ))
plt.imshow(final_array, cmap = 'gray')
plt.show()



import os 
path = os.path.join(current_path,'test_pic')



img_array = cv2.imread(os.path.join(path,'dogs_for_today_2.jpg'),cv2.IMREAD_GRAYSCALE) #
plt.imshow(img_array, cmap = "gray")
plt.show()


new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE ))
plt.imshow(new_array, cmap = 'gray')
plt.show()

x_test_pr = np.array(new_array).reshape(-1,IMG_SIZE, IMG_SIZE,1)
x_pred= model.predict_classes(x_test_pr)
print(x_pred)
