#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.models import Sequential, model_from_json
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
import datetime
import numpy as np
import os
import json
from time import gmtime, strftime
import ast
