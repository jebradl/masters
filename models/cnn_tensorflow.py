## essentially just redoing the data processing file for images specifically
## or not actually we're using a tensorflow tutorial owo

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


feature_set = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/classified_small'
# feature_set = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/img'
data_dir = pathlib.Path(feature_set)

image_count = len(list(data_dir.glob('*/*.png')))
#print(image_count)

batch_size = 32  # 64 for gpu
img_height = 235
img_width = 352

epochs = 20 # 50

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




AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.Rescaling(1./255, offset=0.0)

num_classes = len(class_names)

# adam = adam()


def create_model():

    model = Sequential([
      tf.keras.layers.Rescaling(1./255),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(128, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(), # view
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])  # still v small network


    model.compile(optimizer=adam.Adam(learning_rate=0.005),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


checkpoint_path = "c:/Users/night/Documents/09/school/actual-masters/git/masters/models/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=200)


# create a new model instance
model = create_model()
# model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[cp_callback],
  verbose=1
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


test_path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/test_random/swipe_20.png'

img = tf.keras.utils.load_img(
    test_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # create batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


def get_config(self):

  config = super().get_config().copy()
  config.update({
      'vocab_size': self.vocab_size,
      'num_layers': self.num_layers,
      'units': self.units,
      'd_model': self.d_model,
      'num_heads': self.num_heads,
      'dropout': self.dropout,
  })
  return config

# !mkdir -p saved_model
model.save('c:/Users/night/Documents/09/school/actual-masters/git/masters/models/saved_models/model_v1.h5')
