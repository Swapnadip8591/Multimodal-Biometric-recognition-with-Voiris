# Multimodal Biometric Recognition using Voice and Iris

This project is a multimodal biometric recognition system that identifies individuals using two distinct biometric traits: **voice** and **iris**. By fusing the scores from both modalities, the system provides a more robust and accurate authentication than a single-modality system.

## 📜 Description

The application provides a user-friendly interface to test the recognition system. A user can select a voice sample and an iris image, and the system will identify the individual from a pre-registered database. The core of this project lies in its implementation of two separate recognition pipelines and a score-level fusion mechanism to combine their results.

## ✨ Key Features

* **Multimodal Biometrics**: Utilizes both voice (behavioral) and iris (physiological) traits for enhanced security.
* **Iris Recognition**: Implements a classic iris recognition pipeline:
    * **Segmentation**: Uses Daugman's integro-differential operator to locate the iris and pupil.
    * **Normalization**: Transforms the circular iris region into a fixed-size rectangular block.
    * **Feature Extraction**: Employs Gabor filters to create a unique binary "iris code."
    * **Matching**: Calculates the Hamming distance for matching.
* **Voice Recognition**: Implements a standard speaker recognition system:
    * **Feature Extraction**: Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from voice samples.
    * **Modeling**: Uses Gaussian Mixture Models (GMMs) to create a statistical model of each user's voice.
    * **Matching**: Calculates the log-likelihood score for matching.
* **Score-Level Fusion**: Combines the scores from both modalities using the **weighted sum rule** to make a final, more accurate decision.
* **Simple UI**: A graphical user interface built with Tkinter allows for easy testing and demonstration.

## 🛠️ Technologies Used

* **Python**
* **OpenCV**: For all image processing tasks in the iris recognition pipeline.
* **Librosa**: For audio processing and extracting MFCC features.
* **Scikit-learn**: For implementing Gaussian Mixture Models (GMMs).
* **TensorFlow/Keras**: For building and training neural network models (as seen in the `traditional nn` folder).
* **Tkinter**: For the graphical user interface.
* **NumPy** & **SciPy**: For numerical operations and scientific computing.

## 📊 Datasets

This project is designed to work with the following datasets:

* **Voice**: The [CorporaBangla](https://www.kaggle.com/datasets/arijitx/corporabangla) dataset, which contains Bengali speech data.
* **Iris**: The [DOBES](https://www.kaggle.com/datasets/debranjansarkar/dobes-database) dataset, which is a database of eye images.

*The provided `dataset` folder in this repository contains a sample subset of these datasets.*

## 🚀 How to Run the Project

### 1. Prerequisites

* Python 3.x  
* A C++ compiler (Note: This is only needed if `pip` cannot find pre-compiled packages for your system and needs to build libraries like `scipy` or `scikit-learn` from source. Most users will not need this.)

### 2. Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/multimodal-biometric-recognition-with-voiris.git
cd multimodal-biometric-recognition-with-voiris
```
Install the required Python packages using the requirements.txt file:
```bash
pip install -r requirements.txt
```

### 3. Usage

* First voice_to_raw_image.py will be executed to get the raw images from voice data
* Then scale_n_merge.py will be executed to get the fusion image of raw voice image and iris image
* nn_model.py will be executed to train the model(mapping the data points for nearest neighbour algorithm)
* nn_test.py is used to evaluate the model
