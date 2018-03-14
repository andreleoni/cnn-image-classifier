from import_modules import *

class CNN():
  def __init__(self, name):
    self.classifier = Sequential()
    self.cnn_name = name

  def training_set_path(self):
    return 'dataset/train'

  def test_set_path(self):
    return 'dataset/test'

  def single(self, prediction_url):
    test_image = image.load_img(prediction_url, target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = self.classifier.predict(test_image)

    result_key = result[0][0]
    single_msg = "Single result: " + self.revert_key_to_value(self.class_indices)[result_key]
    print(single_msg)
    self.register_log(single_msg)

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
    class_indices_msg = "The model class indices are:", self.class_indices
    print(class_indices_msg)
    self.register_log(class_indices_msg)

    self.classifier.fit_generator(training_set,
                                  steps_per_epoch = 8000,
                                  epochs = 25,
                                  validation_data = test_set,
                                  validation_steps = 2000)

  def train(self):
    self.learn_layers()
    self.fit_cnn_to_images()
    self.save_model()

  def save_model(self):
    model_backup_path = 'dataset/models/' + self.cnn_name + '.h5'
    self.classifier.save(model_backup_path)

    save_msg = "Model: " + self.cnn_name + " \n Saved to: " + model_backup_path
    print(save_msg)
    self.register_log(save_msg)

  def register_log(self, message):
    log_path = 'train_log.log'
    log_file = open(log_path, 'w+')
    log_file.write('\n')
    log_file.write('######### ' + self.cnn_name + ' #######')
    log_file.write('\n')
    log_file.write(str(message))
    log_file.write('\n')
    log_file.write('#######################################')
    log_file.write('\n')
    log_file.close()

  def revert_key_to_value(self, hash):
    return { v: k for k, v in hash.items() }
