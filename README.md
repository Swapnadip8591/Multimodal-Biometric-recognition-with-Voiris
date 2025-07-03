# Multimodal Biometric recognition with Voiris
 A Multimodal Biometric System implemented using raw level fusion of Voice and Iris traits to recognize human.

## Steps in the project
- The voice is converted to image based on raw file.
- Scaling(or padding) is done to match width of iris and voice images.
- Using a Nearest Neighbour Classifier the model is trained.

### Important parts
The project is built on closed dataset, so no intruder detection has been implemented here.
