#scaling is done by resizing the image to the same width while maintaining aspect ratio
import os
import cv2
import numpy as np
from itertools import product


def resize_image_to_width(img, target_width):
    original_height, original_width = img.shape[:2]
    
    if original_width == target_width:
        return img

    # Compute new height maintaining aspect ratio
    scale_ratio = target_width / original_width
    new_height = int(original_height * scale_ratio)

    # Resize with aspect ratio preserved
    resized_img = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_img


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
            # Get original widths
            width_iris = IrisImage.shape[1]
            width_voice = VoiceImage.shape[1]

            # Determine the smaller width
            target_width = min(width_iris, width_voice)

            # Resize both to the same width
            resized_iris = resize_image_to_width(IrisImage, target_width)
            resized_voice = resize_image_to_width(VoiceImage, target_width)

            # Now merge vertically
            MergedImage = np.vstack((resized_iris, resized_voice))


            # Save merged image
            save_filename = f'class_{class_id}_pair_{i}.png'
            save_path = os.path.join(save_folder, save_filename)
            cv2.imwrite(save_path, MergedImage)

            print(f"Saved: {save_path}")

iris_base_path = '/Desktop/dataset/iris/training'        # e.g., 'dataset/iris'
voice_base_path = '/dataset/raw_image/training'      # e.g., 'dataset/voice'
merged_output_path = '/dataset/merged'  # e.g., 'dataset/merged'

generate_and_save_merged_images(iris_base_path, voice_base_path, merged_output_path)
