import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# model = load_model('./generator_mnist_8.h5')
# model.summary()
# z = np.random.normal(0, 1, (1, 100))
# fake_imgs = model.predict(z)
# fake_imgs = 0.5 * fake_imgs + 0.5
# plt.imshow(fake_imgs.reshape(28, 28))
# plt.show()
# exit()

number_GAN_models = []
for i in range(10):
    try: number_GAN_models.append(load_model('./models/generator_mnist_{}.h5'.format(i)))
    except: number_GAN_models.append(load_model('./models/generator_mnist_8.h5'))
four_digit_number = '4621'
numbers = list(str(four_digit_number))
print(numbers)
imgs = []
for i in numbers:
    i = int(i)
    z = np.random.normal(0, 1, (1, 100))
    fake_imgs = number_GAN_models[i].predict(z)
    fake_imgs = 0.5 * fake_imgs + 0.5
    imgs.append(fake_imgs.reshape(28, 28))
    print(fake_imgs.shape)

img = imgs[0]
for i in range(1,4):
    img = np.append(img, imgs[i], axis=1)

print(img.shape)
plt.cool() # gray, cool, hot, spring, summer, autumn, winter, bone, copper, magma, pink, prism, plasma
plt.imshow(img)
plt.axis('off')
plt.show()

# _, axs = plt.subplots(1, 4, figsize=(10, 40),
#                     sharey=True, sharex=True)
# cnt = 0

# for j in range(4):
#     axs[j].imshow(img[j].reshape(28, 28), cmap='gray')
#     axs[j].axis('off')
#     cnt += 1
# plt.show()


