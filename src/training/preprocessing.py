import os
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define the paths
input_dir = "input/2 Classes"  # Directory containing subfolders for each tumor type
output_dir = "preprocessed"  # Directory to save the preprocessed images
# features_output_file = 'features.npy'  # File to save the extracted features

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the list to store the extracted features
features_list = []

# Iterate over each folder (tumor type) in the input directory
for tumor_type in os.listdir(input_dir):
    tumor_folder = os.path.join(input_dir, tumor_type)
    if os.path.isdir(tumor_folder):
        # Iterate over each image in the tumor folder
        for filename in os.listdir(tumor_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Load the image
                img_path = os.path.join(tumor_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Image Enhancement
                # Histogram equalization
                img_eq = cv2.equalizeHist(img)

                # Median filtering
                img_filtered = cv2.medianBlur(img_eq, 3)

                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                contrast_enhanced_image = clahe.apply(img_filtered)

                # Image Segmentation
                # Thresholding
                _, img_binary = cv2.threshold(img_filtered, 170, 255, cv2.THRESH_BINARY)

                # Texture Analysis using GLCM
                glcm_features = []
                angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
                for angle in angles:
                    # Apply texture analysis using GLCM
                    glcm = cv2.filter2D(
                        img_filtered,
                        -1,
                        np.array(
                            [
                                [np.cos(angle) ** 2, np.sin(angle) * np.cos(angle)],
                                [np.sin(angle) * np.cos(angle), np.sin(angle) ** 2],
                            ]
                        ),
                    )
                    # Calculate GLCM features
                    contrast = np.mean((glcm - np.mean(glcm)) ** 2)
                    dissimilarity = np.mean(np.abs(glcm - np.mean(glcm)))
                    homogeneity = np.mean(1 / (1 + (glcm - np.mean(glcm)) ** 2))
                    energy = np.mean(glcm**2)
                    correlation = np.mean((glcm - np.mean(glcm)) ** 2) / (
                        np.std(glcm) ** 2
                    )
                    asm = np.mean(glcm**2)
                    glcm_features.extend(
                        [contrast, dissimilarity, homogeneity, energy, correlation, asm]
                    )

                # Shape Analysis
                contours, _ = cv2.findContours(
                    img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                shape_features = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    moments = cv2.moments(contour)
                    if area > 0:
                        eccentricity = np.sqrt(
                            (moments["mu20"] - moments["mu02"]) ** 2
                            + 4 * moments["mu11"] ** 2
                        ) / (moments["mu20"] + moments["mu02"] + 2e-16)
                        compactness = (perimeter**2) / area
                    else:
                        eccentricity = 0.0
                        compactness = 0.0
                    shape_features.extend([area, perimeter, eccentricity, compactness])

                # Store the extracted features in a list
                features = np.concatenate([glcm_features, shape_features])
                features_list.append(features)
                contrast_enhanced_image = cv2.resize(
                    contrast_enhanced_image, (224, 224)
                )
                # Save the preprocessed image
                processed_filename = os.path.splitext(filename)[0] + "_processed.jpg"
                processed_path = os.path.join(
                    output_dir, tumor_type, processed_filename
                )
                os.makedirs(os.path.dirname(processed_path), exist_ok=True)
                cv2.imwrite(processed_path, contrast_enhanced_image)

# Convert the features list to a NumPy array
features_array = np.array(features_list)

# Normalize the features array
# scaler = MinMaxScaler()
# normalized_features = scaler.fit_transform(features_array)

# Save the features array
# np.save(features_output_file, normalized_features)
