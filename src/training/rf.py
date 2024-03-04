import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set the path to the directory containing the preprocessed images
data_directory = 'preprocessed'

# Load the preprocessed images and labels
X = []
y = []

for class_name in os.listdir(data_directory):
    class_dir = os.path.join(data_directory, class_name)
    if os.path.isdir(class_dir):
        # Iterate over the images in the class directory
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            # Load the image and preprocess it (replace with your own preprocessing steps)
            image = cv2.imread(image_path) # Load and preprocess the image
            X.append(image)
            y.append(class_name)

# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Reshape the input data if needed
X = X.reshape(X.shape[0], -1)  # Uncomment this line if the input images are flattened

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict labels for the test set
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
classification_rep = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
confusion_mat = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

classification_rep_filtered = {k: v for k, v in classification_rep.items() if isinstance(v, dict)}

# Save classification report to Excel
classification_rep_df = pd.DataFrame.from_dict(classification_rep, orient='index')
classification_rep_df.to_excel('output/classification_report_rf.xlsx')

# Save confusion matrix
confusion_mat_df = pd.DataFrame(confusion_mat, index=label_encoder.classes_, columns=label_encoder.classes_)
confusion_mat_df.to_excel('output/confusion_matrix_rf.xlsx')

# Save F1 score to a text file
with open('output/f1_score_rf.txt', 'w') as f:
    f.write(f'F1 Score: {f1}')
