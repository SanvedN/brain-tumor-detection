import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

# Load the saved CNN model
model = load_model('models/cnn_model.h5')

# Load and preprocess the test image
test_image_path = 'path_to_img'
img = cv2.imread(test_image_path)
img = cv2.resize(img , (128,128))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_array = np.asarray(img)
img_array = np.reshape(img_array, (1, 128, 128, 3))

# Make predictions on the test image
predictions = model.predict(img_array)
type = np.argmax(predictions, axis=1)
print(type)