#scaling is done by padding zeros to the right side of the image
import os
import cv2
import numpy as np
from itertools import product

# Padding function (right-pad with zeros)
def pad_image_to_width(img, target_width):
    height, width = img.shape[:2]
    
    # Ensure image has 3 channels
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    
    padding = target_width - width
    if padding > 0:
        padded_img = np.pad(img, ((0, 0), (0, padding), (0, 0)), mode='constant')
    else:
        padded_img = img
    return padded_img

# Load image in color
def load_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

# Generate all merged images and save
def generate_and_save_merged_images(iris_base_path, voice_base_path, merged_output_path):
    for class_id in range(1, 9):
        iris_folder = os.path.join(iris_base_path, str(class_id))
        voice_folder = os.path.join(voice_base_path, str(class_id))
        save_folder = os.path.join(merged_output_path, str(class_id))

        # Create class subfolder in output directory
        os.makedirs(save_folder, exist_ok=True)

        # List image files
        iris_images = [os.path.join(iris_folder, f) for f in os.listdir(iris_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        voice_images = [os.path.join(voice_folder, f) for f in os.listdir(voice_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Cartesian product of iris and voice images
        for i, (iris_path, voice_path) in enumerate(product(iris_images, voice_images)):
            IrisImage = load_image(iris_path)
            VoiceImage = load_image(voice_path)

            if IrisImage is None or VoiceImage is None:
                print(f"Skipping failed load: {iris_path}, {voice_path}")
                continue

            # Determine target width
            target_width = max(IrisImage.shape[1], VoiceImage.shape[1])

            # Pad and merge
            padded_iris = pad_image_to_width(IrisImage, target_width)
            padded_voice = pad_image_to_width(VoiceImage, target_width)
            MergedImage = np.vstack((padded_iris, padded_voice))

            # Save merged image
            save_filename = f'class_{class_id}_pair_{i}.png'
            save_path = os.path.join(save_folder, save_filename)
            cv2.imwrite(save_path, MergedImage)

            print(f"Saved: {save_path}")

iris_base_path = '/dataset/iris/test'        # e.g., 'dataset/iris'
voice_base_path = '/dataset/raw_image/test'      # e.g., 'dataset/voice'
merged_output_path = '/dataset/merged_pad/testing'  # e.g., 'dataset/merged'

generate_and_save_merged_images(iris_base_path, voice_base_path, merged_output_path)
