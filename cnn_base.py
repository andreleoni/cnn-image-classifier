from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
import numpy as np
import os

class CNN_Base(object):
  def __init__(self, name):
    self.classifier = Sequential()
    self.cnn_name = name

  def set_class_indices(self, class_indices):
    self.class_indices = class_indices

  def classifier_add(self, classifier_attr):
    self.classifier.add(classifier_attr)

  def classifier_compile(self, *args, **kwargs):
    self.classifier.compile(*args, **kwargs)

  def classifier_fit_generator(self, *args, **kwargs):
    self.classifier_fit_generator(*args, **kwargs)

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
    self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    self.register_log("Loaded model: " + name)
    self.evaluate_model()

  def evaluate_model(self):
    self.register_log("%s: %.2f%%" % (self.classifier.metrics_names[1], self.classifier.evaluate()[1]*100))
