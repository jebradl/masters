import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

model = tf.keras.models.load_model('c:/Users/night/Documents/09/school/actual-masters/git/masters/models/saved_models/model_v1.h5')
model.summary()