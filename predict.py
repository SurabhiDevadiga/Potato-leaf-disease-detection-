import tensorflow as tf
import cv2
import numpy as np

model=tf.keras.models.load_model(
"potato_model.h5"
)

classes=[
"Early Blight",
"Late Blight",
"Healthy"
]

img=cv2.imread(
r"C:\Users\SURABHI\PotatoDiseaseDetection\test.JPG"
)

if img is None:
    print("Image not found")
    exit()

img=cv2.resize(img,(128,128))

img=img/255.0

img=np.expand_dims(img,axis=0)

prediction=model.predict(img)

result=np.argmax(prediction)

print("Prediction:",classes[result])