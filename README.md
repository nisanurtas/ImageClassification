# ImageClassification
image classifier using MobileNetV3Large for 4 classes (cat, empty, human, vehicle). Easy customization for diverse datasets.


This repository contains a Python script for training a convolutional neural network (CNN) using the MobileNetV3Large architecture for image classification. The trained model can classify images into one of four classes: "cat," "empty," "human," or "vehicle."

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [For testing](#for-testing)



## Getting Started

Clone this repository to your local machine to get started.

## Prerequisites:
Python 3.x
TensorFlow
Keras
OpenCV (if you plan to use it, otherwise, you can remove the cv2.waitKey(1) line from the code)
You can install the required packages using pip:

pip install tensorflow keras opencv-python

## Usage:
You can use this script to train an image classification model for your specific dataset. Follow the steps below:

1-Organize your dataset into training and validation sets in separate directories.

2-Update the train_path and valid_path variables in the script to point to your dataset directories.

3-Customize the classes_train_valid variable to match your specific class labels.

4-Run the script:
python train_classification_model.py


## For testing: ##

## Inference
1-Load the pre-trained model and open a video stream.

2-Process video frames, resize them, and use the model to classify them in real-time.

3-Display classification results on the video feed along with frames per second (FPS).

4-Run the inference script:

python classify_video.py

