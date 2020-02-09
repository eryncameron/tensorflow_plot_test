import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images / 255.0

test_images = test_images / 255.0

"""print(train_labels[0])
This part I got to work
"""
class_names = ['T-shirt/top', 'Trouse', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.imshow(train_images[1], cmap=plt.cm.binary)
plt.show()
print(test_labels)

