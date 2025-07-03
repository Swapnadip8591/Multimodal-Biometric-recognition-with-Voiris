import math
import numpy as np
import cv2
import librosa
import soundfile as sf
import os

# Function to fix the length of audio to a specified duration in seconds
def fix_audio_length(audio, sr, m):
    target_length = int(sr * m)
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        repeats = int(np.ceil(target_length / len(audio)))
        audio = np.tile(audio, repeats)
        return audio[:target_length]
    else:
        return audio

def list_files_in_directory(directory_path):
    try:
        # Get a list of all files and directories in the specified path
        items = os.listdir(directory_path)
        
        # Filter to include only files
        files = [item for item in items if os.path.isfile(os.path.join(directory_path, item))]
        
        return files
    except FileNotFoundError:
        print(f"The directory '{directory_path}' does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied for accessing '{directory_path}'.")
        return []

# making a rgb image with the data
def list_to_rgb_image(lst):
    # Pad with zeros if necessary to make the length divisible by 3
    remainder = len(lst) % 3
    if remainder != 0:
        padding = [0] * (3 - remainder)
        lst += padding

    total_pixels = len(lst) // 3

    # Find height and width such that height * width = total_pixels
    factors = [(i, total_pixels // i) for i in range(1, int(math.sqrt(total_pixels)) + 1) if total_pixels % i == 0]
    if not factors:
        height, width = 1, total_pixels  # fallback if total_pixels is prime
    else:
        height, width = min(factors, key=lambda x: abs(x[0] - x[1]))

    # Reshape to (height, width, 3)
    arr = np.array(lst).reshape((height, width, 3))
    return arr

directory_path= "/dataset/audio/m2_a4/"  #paths are added manually here
output_audio= "/audio/7secaud.wav"  #temporary output file
img_directory= "/dataset/raw_image/m2_a4/"
files = list_files_in_directory(directory_path)

if files:
    for file1 in files:
        file= directory_path+file1
        # Load the audio file
        if file1.endswith(".wav"):
            # Load the audio
            audio, sr = librosa.load(file, sr=22040)

            # Fix the length
            m = 7  # Desired duration in seconds
            fixed_audio = fix_audio_length(audio, sr, m)

            # Save the output
            
            sf.write(output_audio, fixed_audio, sr)

            li=[]
            with open(output_audio, "rb") as audio_file:
                byte_data = audio_file.read()

                #decode the binary data
                decoded_text = byte_data.decode('latin1')
                decoded_text= decoded_text + "flag"

                for char in decoded_text:
                    li.append(ord(char))

            img = list_to_rgb_image(li)
            print("Shape:", img.shape)
            # print(img)
            img_name,_= file1.split(".")
            image= img_directory+img_name+".png"
            cv2.imwrite(image, img)

# #checking if image is same as data
# img_read= cv2.imread(image)
# # print(img_read)
# if np.array_equal(img, img_read):
#     print("equal")

# #removing flag(code part for reverse)
# byte_ascii= img_read.flatten()
# data=""
# for x in byte_ascii:
#    data+=chr(x)

# flag_index= data.find("flag:")
# data= data[:flag_index]


# binary_data = data.encode('latin1')
# reconstructed="D:/msc_project/audio/f1_a2.wav"
# with open(reconstructed, "wb") as audio_file:
#     audio_file.write(binary_data)