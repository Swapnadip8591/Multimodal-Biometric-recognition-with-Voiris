import os
import cv2
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Load training data
def load_images_from_folder(folder_path):
    X = []
    y = []
    for label in sorted(os.listdir(folder_path)):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                X.append(img.flatten())
                y.append(int(label))
    return np.array(X), np.array(y)

X_train_raw, y_train = load_images_from_folder("/dataset/merged_pad/training")
X_test_raw, y_test = load_images_from_folder("/dataset/merged_pad/testing")

# PCA (no dimensionality reduction, just decorrelation)
pca = PCA()
X_train = pca.fit_transform(X_train_raw)
X_test = pca.transform(X_test_raw)

# 1-NN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Clean up PCA model to save space
light_pca = PCA()
light_pca.components_ = pca.components_
light_pca.mean_ = pca.mean_
light_pca.n_components_ = pca.n_components_
light_pca.explained_variance_ = pca.explained_variance_
light_pca.explained_variance_ratio_ = pca.explained_variance_ratio_

# Save model and PCA 
joblib.dump(knn, "/models/knn_model.joblib")
joblib.dump(light_pca, "/models/pca_model.joblib")

print("Model and PCA saved.")