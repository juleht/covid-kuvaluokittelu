import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy





# Aineiston muokkaus ja lataaminen
polku_training = 'src/Covid19-dataset/train'
muoto = 'categorical'
vari = 'grayscale'
koko = (256,256)
batch = 32


training_data_generator = ImageDataGenerator(
  rescale = 1.0/255,
  zoom_range = 0.1,
  rotation_range = 25,
  width_shift_range = 0.05,
  height_shift_range = 0.05
)

validation_data_generator = ImageDataGenerator()

training_iterator = training_data_generator.flow_from_directory(
  polku_training,
  class_mode = muoto,
  color_mode = vari,
  batch_size = batch
)


validation_iterator = validation_data_generator.flow_from_directory(
  directory = polku_training,
  class_mode = muoto,
  color_mode = vari,
  batch_size = batch
)

sample_batch_input, sample_batch_labels  = training_iterator.next()
 
print(sample_batch_input.shape,sample_batch_labels.shape)


# Mallin kokoaminen

malli = Sequential()
malli.add(tf.keras.Input(shape=(256,256, 1)))
malli.add(layers.Conv2D(5, 5,strides = 2, activation ='relu'))
malli.add(layers.MaxPooling2D(pool_size=(2, 2),  strides=(2, 2)))
malli.add(layers.Dropout(0.1))
malli.add(layers.Conv2D(5, 5, strides = 1, activation ='relu'))
malli.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
malli.add(layers.Dropout(0.2))
malli.add(layers.Flatten())
malli.add(layers.Dense(3, activation = 'softmax'))
malli.summary()


# Mallin sovittaminen
malli.compile(
  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
  loss = tf.keras.losses.CategoricalCrossentropy(),
  metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()],
)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
malli.fit(
  training_iterator,
  steps_per_epoch=training_iterator.samples/batch,
  epochs=100,
  validation_data=validation_iterator,
  validation_steps=validation_iterator.samples/batch,
  callbacks=[es]
  )


print('Accuracy: ', list(malli.history.history.items())[1][-1][-1])
print('AUC: ', list(malli.history.history.items())[2][-1][-1])

# Mallin tulosten arviointi
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(malli.history.history['categorical_accuracy'])
ax1.plot(malli.history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'])

ax2 = fig.add_subplot(2,1,2)
ax2.plot(malli.history.history[list(malli.history.history.keys())[2]])
ax2.plot(malli.history.history[list(malli.history.history.keys())[-1]])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'])

fig.tight_layout()
plt.show()

test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predictions = malli.predict(validation_iterator, steps=test_steps_per_epoch)
test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predicted_classes = numpy.argmax(predictions, axis=1)
true_classes = validation_iterator.classes
class_labels = list(validation_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   
 
cm=confusion_matrix(true_classes,predicted_classes)
print(cm)
