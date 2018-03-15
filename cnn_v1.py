from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
import numpy as np
import os

from cnn_base import *

class CNN_v1(CNN_Base):
  def learn_layers(self):
    # Step 1 - Convolution
    self.classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128 , 3), activation = 'relu'))

    # Step 2 - Pooling
    self.classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    self.classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    self.classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a third convolutional layer
    self.classifier.add(Conv2D(64, (3, 3), activation='relu'))
    self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 3 - Flattening
    self.classifier.add(Flatten())

    # Step 4 - Full connection
    self.classifier.add(Dense(units = 128, activation = 'relu'))
    self.classifier.add(Dropout(0.5))
    self.classifier.add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling the CNN
    self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

  def fit_cnn_to_images(self):
    # Part 2 - Fitting the CNN to the images
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(self.training_set_path(),
                                                    target_size = (128, 128),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

    test_set = test_datagen.flow_from_directory(self.test_set_path(),
                                                target_size = (128, 128),
                                                batch_size = 32,
                                                class_mode = 'binary')

    self.class_indices = training_set.class_indices
    self.register_log("The model class indices are: " + str(self.class_indices))

    self.classifier.fit_generator(training_set,
                                  steps_per_epoch = 8000,
                                  epochs = 25,
                                  validation_data = test_set,
                                  validation_steps = 2000)
