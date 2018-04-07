#!/usr/bin/python
# -*- coding: utf-8 -*-

from import_modules import *

class CNN():
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
                                  epochs = 2,
                                  validation_data = test_set,
                                  validation_steps = 2000)

  def train(self):
    self.learn_layers()
    self.fit_cnn_to_images()
    self.save_model()

  def save_model(self):
    self.save_class_indices()
    self.save_as_json()
    self.save_as_h5()

  def save_class_indices(self):
    with open('dataset/models/' + self.cnn_name + '_class_indices.json', "w") as json_file:
      json_file.write(json.dumps(self.class_indices, ensure_ascii=False))

  def save_as_json(self):
    model_json = self.classifier.to_json()
    with open('dataset/models/' + self.cnn_name + '.json', "w") as json_file:
      json_file.write(model_json)

  def save_as_h5(self):
    model_backup_path = os.path.join(self.script_dir(), 'dataset/models/' + self.cnn_name + '.h5')
    self.classifier.save_weights(model_backup_path)
    self.register_log("Model: " + self.cnn_name + " \n Saved to: " + model_backup_path)

  def register_log(self, message):
    print(message)

    log_path = os.path.join(self.script_dir(), 'train_log.log')
    log_file = open(log_path, 'w+')
    log_file.write('#> ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '  ' + self.cnn_name + ' : ' + str(message))
    log_file.write('\n')
    log_file.close()

  def revert_key_to_value(self, hash):
    return { v: k for k, v in hash.items() }

  def load_model(self):
    self.load_class_indices()
    self.learn_layers()
    self.classifier.load_weights('dataset/models/' + self.cnn_name + '.h5')
    self.register_log("Loaded model: " + self.cnn_name)

  def load_class_indices(self):
    json_file = open('dataset/models/' + self.cnn_name + '_class_indices.json', 'r')
    self.class_indices = ast.literal_eval(json_file.read())
    json_file.close()

  def evaluate_model(self):
    self.register_log("%s: %.2f%%" % (self.classifier.metrics_names[1], self.classifier.evaluate()[1]*100))
