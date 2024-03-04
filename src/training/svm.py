import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
import matplotlib.pyplot as plt
import pandas as pd

# Set the data directory path
data_directory = "preprocessed"

# Create a list to hold the data and labels
data = []
labels = []

# Iterate over the subdirectories (classes) in the data directory
for class_name in os.listdir(data_directory):
    class_dir = os.path.join(data_directory, class_name)
    if os.path.isdir(class_dir):
        # Iterate over the images in the class directory
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            # Load the image and preprocess it (replace with your own preprocessing steps)
            image = cv2.imread(image_path)  # Load and preprocess the image
            data.append(image)
            labels.append(class_name)

# Convert the data and labels lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Shuffle the data and labels in unison
random_indices = np.random.permutation(len(data))
data = data[random_indices]
labels = labels[random_indices]

# Split the data into training and testing sets (adjust the split ratio as needed)
split_ratio = 0.8
split_index = int(split_ratio * len(data))

X_train = data[:split_index]
y_train = labels[:split_index]
X_test = data[split_index:]
y_test = labels[split_index:]


# Reshape the data if needed (uncomment and modify the line below)
# X_train = X_train.reshape(-1, image_height, image_width, num_channels)
# X_test = X_test.reshape(-1, image_height, image_width, num_channels)

# Flatten the image data if needed (uncomment and modify the line below)
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# Train the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = svm_model.predict(X_test)

# Calculate evaluation metrics
classification_rep = classification_report(y_test, y_pred, zero_division=0)
confusion_mat = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

# Save classification report to Excel
classification_rep = classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(classification_rep).transpose()
report_df.to_excel("output/svm_classification_report.xlsx", index=True)

# Save confusion matrix visualization
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
disp.plot()
plt.title("SVM Confusion Matrix")
plt.savefig("output/svm_confusion_matrix.png")
plt.close()

# Save F1 score to a text file
with open("output/svm_f1_score.txt", "w") as f:
    f.write(f"SVM F1 Score: {f1}")
