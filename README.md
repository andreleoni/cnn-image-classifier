# Convolutional Neural Network - Image Predictions

### This project uses

* Install Theano
  `pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git`

* Install Tensorflow
  `https://www.tensorflow.org/install/`

* Install Keras
  `pip install --upgrade keras`

# Instructions

## Creathe the tree in the root of object

```
| dataset
|-- models
|-- single
|---- photo.jpg
|-- test
|---- folder_1
|---- folder_2
|-- train
|---- folder_1
|---- folder_2
```

In the folder_1 , folder_2, put your images, from test and from train.

I recommend, test be at least the 25% of the train images.

The test image will send you a first feedback about the accurace in your learn algorithm with your train dataset.

#####

Example train dataset with `cats` and `dogs` images:

`https://drive.google.com/file/d/1uWTJ4nogWaRukzkwfLOfSRSj9ByvNFeb/view?usp=sharing`


## How to RUN

`docker run -it -v ${PWD}:/data andreleoni/cnn-tensorflow `
`shell# python3`

On python, import CNN file, instance, and execute than
```
from cnn import *
cnn = CNN('my_cnn')
cnn.train()
```
