import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model

model = load_model('./models/GAN_mnist_0.h5')
model.summary()
exit()


number_GAN_models = []
for i in range(10):
    number_GAN_models.append(load_model('./models/GAN_mnist_{}.h5'.format(i)))

four_digit_number = 1234
numbers = list(str(four_digit_number))
print(numbers)
img = []
for i in numbers:
    i = int(i)
    z = np.random.normal(0, 1, (4 * 4, 100))
    img.append((number_GAN_models[i].predict(z)))
_, axs = plt.subplots(1, 4, figsize=(1, 4),
                    sharey=True, sharex=True)
cnt = 0

for j in range(4):
    axs[i, j].imshow(img[cnt, :, :, 0], cmap='gray')
    axs[i, j].axis('off')
    cnt += 1
plt.show()




