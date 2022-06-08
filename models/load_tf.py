import os
import PIL
import pathlib
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2 import adam


feature_set = 'C:/Users/jeb1618/masters/models/data/fma/classified_small'
data_dir = pathlib.Path(feature_set)

image_count = len(list(data_dir.glob('*/*.png')))

img_height = 235
img_width = 352

batch_size = 64
epochs = 10


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = keras.models.load_model('C:/Users/jeb1618/masters/models/saved_models/model_v3')
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs,
#   verbose=1
# )

loss, acc = model.evaluate(val_ds)


def get_scores(path):
    # change img size and not having to save then load img?
    # also scores=None type beat
    
    img = tf.keras.utils.load_img(
        path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    scores = []
    for i in score:
        scores.append(float(i))
    
    return scores


print(get_scores('C:/Users/jeb1618/masters/models/data/fma/test_random/swipe_20.png'))