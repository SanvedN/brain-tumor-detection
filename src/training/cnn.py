import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(17, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    'preprocessed',
    labels='inferred',
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(128, 128),
    batch_size=32,
    label_mode = 'categorical'
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    'preprocessed',
    labels='inferred',
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(128, 128),
    batch_size=32,
    label_mode = 'categorical'
)

# Train the model
history = model.fit(train_data, validation_data=val_data, epochs=35)

# Evaluate the model
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'preprocessed',
    labels='inferred',
    image_size=(128, 128),
    batch_size=32,
    label_mode='categorical'
)
loss, accuracy = model.evaluate(test_data)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
model.save('cnn_model.h5') #97% accuracy

# Predict the labels for the test data
y_true = []
y_pred = []

for images, labels in test_data:
    y_true.extend(tf.argmax(labels, axis=1).numpy())
    y_pred.extend(tf.argmax(model.predict(images), axis=1).numpy())

# Calculate evaluation metrics
print("Classification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("F1 Score:")
print(f1_score(y_true, y_pred, average='macro'))

# Calculate evaluation metrics
classification_rep = classification_report(y_true, y_pred)
confusion_mat = confusion_matrix(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

# Save classification report to Excel
classification_rep = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(classification_rep).transpose()
report_df.to_excel(f'output/classification_report.xlsx', index=True)

# Save confusion matrix visualization
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
disp.plot()
plt.title(f'Confusion Matrix')
plt.savefig(f'output/confusion_matrix.png')
plt.close()

# Save F1 score to a text file
with open(f'output/f1_score.txt', 'w') as f:
    f.write(f'F1 Score')


#plotting accuracy and loss
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'],loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc = 'upper left')
plt.show()