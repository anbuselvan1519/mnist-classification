# Convolutional Deep Neural Network for Digit Classification.
### Name: Anbuselvan.S
### Reference No: 212223240008

## AIM:
To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset:
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model:
![image](https://github.com/anbuselvan1519/mnist-classification/assets/139841744/4a04c4e9-3753-4448-83c4-5b8e541a0c7a)
## DESIGN STEPS:

### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Build a CNN model

### STEP 3:
Compile and fit the model and then predict

## PROGRAM:

### Name:Anbuselvan.S
### Register Number:212223240008

#### Library Importing:
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
```

#### Shaping:
```
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
```

#### One Hot Encoding:
```
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```

#### CNN Model:
```
model = keras.Sequential()
# Write your code here
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(layers.Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(15,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
```

#### Metrics:
```
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```

#### Prediction:
```
img = image.load_img('imagefive.jpeg')
type(img)
img = image.load_img('imagefive.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
```

## OUTPUT:

### Training Loss, Validation Loss Vs Iteration Plot:

![image](https://github.com/anbuselvan1519/mnist-classification/assets/139841744/2798d309-1f04-4c91-a316-d225bfa25650)
![image](https://github.com/anbuselvan1519/mnist-classification/assets/139841744/e9b76c85-ebfb-49b0-a8fd-3131c67173a0)

### Classification Report

![image](https://github.com/anbuselvan1519/mnist-classification/assets/139841744/b2b91122-9187-4e3e-8500-f07364ee1ccd)

### Confusion Matrix

![image](https://github.com/anbuselvan1519/mnist-classification/assets/139841744/7bd1919e-a1e3-4ecd-bb25-81c6706af464)

### New Sample Data Prediction

![image](https://github.com/anbuselvan1519/mnist-classification/assets/139841744/08557d2c-45c5-473b-9c33-a335c2d0e0bc)

## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
