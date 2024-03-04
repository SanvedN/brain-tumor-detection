import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

# Load the saved CNN model
model = load_model('models/2_classes_128_128.h5')

# Load and preprocess the test image
test_image_path = 'path_to_img'
img = cv2.imread(test_image_path)
img = cv2.resize(img , (128,128))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
filtered_image = cv2.medianBlur(img, 5)
# Enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast_enhanced_image = clahe.apply(filtered_image)
# Perform edge detection
edges = cv2.Canny(contrast_enhanced_image, 30, 150)
# Perform image segmentation
_, thresholded_image = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('img',thresholded_image)
cv2.waitKey(0)
img = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2RGB)
img_array = np.asarray(img)
img_array = np.reshape(img_array, (1, 128, 128, 3))

# Make predictions on the test image
predictions = model.predict(img_array)
type = np.argmax(predictions, axis=1)
print(type)