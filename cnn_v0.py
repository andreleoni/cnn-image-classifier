from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from cnn_base import *

class CNN_v0(CNN_Base):
  def __init__(self, name):
    super().__init__(name)

  def train(self):
    self.learn_layers()
    self.fit_cnn_to_images()
    self.save_model()

  def learn_layers(self):
    # Step 1 - Convolution
    super().classifier_add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

    # Step 2 - Pooling
    super().classifier_add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    super().classifier_add(Flatten())

    # Step 4 - Full connection
    super().classifier_add(Dense(units = 128, activation = 'relu'))
    super().classifier_add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling the CNN
    super().classifier_compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

  def fit_cnn_to_images(self):
    # Part 2 - Fitting the CNN to the images
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(super().training_set_path(),
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

    test_set = test_datagen.flow_from_directory(super().test_set_path(),
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

    super().set_class_indices(training_set.class_indices)
    super().register_log("The model class indices are:" + str(training_set.class_indices))

    super().classifier_fit_generator(training_set,
                                  steps_per_epoch = 8000,
                                  epochs = 25,
                                  validation_data = test_set,
                                  validation_steps = 2000)
