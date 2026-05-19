import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE=128
BATCH_SIZE=32

train_datagen=ImageDataGenerator(

    rescale=1./255,

    rotation_range=30,

    width_shift_range=.2,

    height_shift_range=.2,

    zoom_range=.2,

    shear_range=.2,

    horizontal_flip=True,

    brightness_range=[0.8,1.2],

    validation_split=.2
)

train_generator=train_datagen.flow_from_directory(

    "dataset",

    target_size=(IMG_SIZE,IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='training'
)

val_generator=train_datagen.flow_from_directory(

    "dataset",

    target_size=(IMG_SIZE,IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='validation',

    shuffle=False
)

print(train_generator.class_indices)

model=Sequential()

model.add(
Conv2D(
32,
(3,3),
activation='relu',
input_shape=(128,128,3)
))

model.add(
MaxPooling2D(2,2)
)

model.add(
Conv2D(
64,
(3,3),
activation='relu'
))

model.add(
MaxPooling2D(2,2)
)

model.add(
Conv2D(
128,
(3,3),
activation='relu'
))

model.add(
MaxPooling2D(2,2)
)

model.add(
Flatten()
)

model.add(
Dense(
256,
activation='relu'
)
)

model.add(
Dropout(.5)
)

# Automatic 4 classes
model.add(
Dense(
train_generator.num_classes,
activation='softmax'
)
)

model.compile(

optimizer='adam',

loss='categorical_crossentropy',

metrics=['accuracy']
)

early=EarlyStopping(
monitor='val_loss',
patience=3,
restore_best_weights=True
)

history=model.fit(

train_generator,

validation_data=val_generator,

epochs=20,

callbacks=[early]
)

model.save(
"potato_model.h5"
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.legend(
['Train','Validation']
)

plt.title(
"Accuracy"
)

plt.show()

pred=model.predict(
val_generator
)

pred=np.argmax(
pred,
axis=1
)

true=val_generator.classes

cm=confusion_matrix(
true,
pred
)

sns.heatmap(
cm,
annot=True,
fmt='d'
)

plt.show()

print(

classification_report(

true,

pred,

target_names=list(
train_generator.class_indices.keys()
)
))