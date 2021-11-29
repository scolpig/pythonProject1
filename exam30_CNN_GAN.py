import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

OUT_DIR = './CNN_OUT_img/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
img_shape = (28, 28, 1)
epoch = 100000
batch_size = 128
noise = 100
sample_interval = 100

#buile generator
generator_model = Sequential()
generator_model.add(Dense(256*7*7, input_dim=noise))
generator_model.add(Reshape((7, 7, 256)))
generator_model.add(Conv2DTranspose(128, kernel_size=3,
                strides=2, padding='same'))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))
generator_model.add(Conv2DTranspose(64, kernel_size=3,
                strides=2, padding='same'))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))
generator_model.add(Conv2DTranspose(1, kernel_size=3,
                strides=2, padding='same'))
generator_model.add(Activation('tanh'))
generator_model.summary()

# build discriminator
discriminator_model = Sequential()
discriminator_model.add(Conv2D(32, kernel_size=3,
                strides=2, padding='same', input_shape=img_shape))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Conv2D(64, kernel_size=3,
                strides=2, padding='same'))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Conv2D(128, kernel_size=3,
                strides=2, padding='same'))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Flatten())
discriminator_model.add(Dense(1, activation='sigmoid'))
discriminator_model.summary()

discriminator_model.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy'])
discriminator_model.trainable = False




