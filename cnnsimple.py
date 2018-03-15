from import_modules import *

class CNNSimple():
  def __init__(self, name):
    self.classifier = Sequential()
    self.cnn_name = name

  def script_dir(self):
    return os.path.dirname(__file__)

  def training_set_path(self):
    return os.path.join(self.script_dir(), 'dataset/train')

  def test_set_path(self):
    return os.path.join(self.script_dir(), 'dataset/test')

  def single(self, prediction_url):
    test_image = image.load_img(prediction_url, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = self.classifier.predict(test_image)

    result_key = result[0][0]

    self.register_log("Single result: " + str(self.revert_key_to_value(self.class_indices)[result_key]))

  def learn_layers(self):
      # Step 1 - Convolution
      self.classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

      # Step 2 - Pooling
      self.classifier.add(MaxPooling2D(pool_size = (2, 2)))

      # Step 3 - Flattening
      self.classifier.add(Flatten())

      # Step 4 - Full connection
      self.classifier.add(Dense(units = 128, activation = 'relu'))
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
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

    test_set = test_datagen.flow_from_directory(self.test_set_path(),
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

    self.class_indices = training_set.class_indices
    self.register_log("The model class indices are:" + str(self.class_indices))

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
    model_backup_path = os.path.join(script_dir, 'dataset/models/' + self.cnn_name + '.h5')
    self.classifier.save_weights(model_backup_path)
    self.register_log("Model: " + self.cnn_name + " \n Saved to: " + model_backup_path)

  def register_log(self, message):
    print(message)

    log_path = os.path.join(self.script_dir(), 'train_log.log')
    log_file = open(log_path, 'w+')
    log_file.write('#> ' + self.cnn_name + ' : ' + str(message))
    log_file.write('\n')
    log_file.close()

  def revert_key_to_value(self, hash):
    return { v: k for k, v in hash.items() }

  def load_model(self, name):
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    self.classifier = model_from_json(loaded_model_json)
    self.classifier.load_weights(name + '.h5')
    self.register_log("Loaded model: " + name)
    self.evaluate_model()

  def evaluate_model(self):
    self.register_log("%s: %.2f%%" % (self.classifier.metrics_names[1], self.classifier.evaluate()[1]*100))
