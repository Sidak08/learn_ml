# notes
# nural networks are made of layers
# input layer takes in the raw data
# as many output nurons as the classes of the data we can have
# we also have hidden layers
# if every node connects to the hidden layer then it is densly connected
# connections are called weights
# wieghts are the values that get changed
# there are also biases
# biases have connections which remain at 1
# biases are constansts but programble values
# now we have some complecated math
# there are a few activation functions
# relu rectified linear unit takes any values less then zero and makes it zero
# tanh hyperbolic tangent squishes the values between -1 and 1
# sigmoid squishes the values between 0 and 1
# we use activation functions to move data into a higher demension
# A loss function calculates how far is our output form the actual output
# higher losser function output leads to more changes
# there are different loss functions
# commons one are mean squared error mean absolute error and hinge loss
# gradient descent is used to lead the function to the minimum
# back propogation updates weights and biases as move using gradient descent
# optemizer is the algorithm that does everything

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training

train_images.shape
train_images[0,23,23]  # let's have a look at one pixel

train_labels[:10]  # let's have a look at the first 10 training labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)  # we pass the data, labels and epochs and watch the magic!

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

predictions[0]

np.argmax(predictions[0])

test_labels[0]

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
