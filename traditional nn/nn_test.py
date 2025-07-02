import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA

# === Load test data ===
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

# Confusion Matrix 
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(1, 9))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(1, 9), yticklabels=range(1, 9))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
# PCA Projection of Test Set
def visualize_pca_projection(X, y, title='PCA Projection'):
    pca_2d = PCA(n_components=2)
    X_proj = pca_2d.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_proj[:, 0], y=X_proj[:, 1], hue=y, palette='tab10', s=60)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Class')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Load saved model and PCA 
knn = joblib.load("/models/knn_model.joblib")
pca = joblib.load("/models//pca_model.joblib")

# Load test data
X_test_raw, y_test = load_images_from_folder("/dataset/merged_pad/testing")
X_test = pca.transform(X_test_raw)

# Predict
y_pred = knn.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred)

# Visualize PCA projection of test set
visualize_pca_projection(X_test, y_test, title="Test Set PCA Projection")
