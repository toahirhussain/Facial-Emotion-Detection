## Check the app : https://facialemotiondetector.streamlit.app/
## App UI below
<img width="618" height="1142" alt="{C8B02DDC-EC63-4624-A99F-B197CA84466C}" src="https://github.com/user-attachments/assets/2eb17aaa-71a9-4251-a5d5-dbd02cfc570c" />


# Emotion Detection Using Image Classification (Happy vs Sad)

## 📌 Project Overview

> This project implements a computer vision–based machine learning model to classify facial images into two emotional states: Happy or Sad. The objective is to explore how deep learning techniques can learn visual patterns associated with human emotions and accurately classify them from images.

> The model was trained using images stored locally and demonstrates an end-to-end workflow including data preprocessing, model training, evaluation, and prediction.

## 🎯 Objectives

 - Build an image classification model to distinguish between happy and sad facial expressions
 
 - Preprocess and normalize image data for training
 
 - Train and evaluate a deep learning model
 
 - Demonstrate emotion prediction on unseen images

## 🗂️ Dataset Description

 - Classes: Happy, Sad
 - Trained the model using approximately 300 labeled images.
 
 - Data Source: Images stored locally on the author’s machine
 
 - Image Format: RGB images
 
 - Preprocessing Steps:
 
 - Image resizing
 
 - Normalization
 
 - Label encoding

⚠️ The dataset is not included in this repository due to size and privacy considerations.

## 🧠 Model Architecture

 - Convolutional Neural Network (CNN)
 
 - Multiple convolution and pooling layers
 
 - Fully connected dense layers
 
 - Softmax / Sigmoid activation for classification

## 🔧 Project Workflow

 1️⃣ Data Preparation

  - Loaded images from local directories
  
  - Resized images to a consistent shape
  
  - Normalized pixel values
  
  - Split data into training and validation sets

 2️⃣ Model Training

  - Trained the CNN using labeled image data
  
  - Optimized using appropriate loss function and optimizer
  
  - Monitored training and validation performance

  3️⃣ Model Evaluation

  - Evaluated accuracy on validation data
  
  - Tested predictions on unseen images

  4️⃣ Prediction

  - Classified new images as Happy or Sad
  
  - Output predicted label with confidence score

## 📊 Results
  
  - The model successfully learned visual patterns associated with facial expressions
  
  - Achieved strong classification performance on validation data
  -  Demonstrated reliable predictions on new images

## 🔮 Future Improvements

  - Expand dataset with more emotions (angry, surprised, neutral)
  
  - Apply data augmentation to improve generalization
  
  - Experiment with transfer learning (ResNet, MobileNet)
  
  - Deploy model as a web app using Flask or FastAPI
  
  - Integrate real-time emotion detection using webcam input

## Conclusion

> The emotion classification model successfully shows how deep learning techniques can be applied to human-centric visual tasks such as emotion recognition. While the current implementation focuses on binary classification, the results establish a strong foundation for more advanced emotion detection systems. With additional data, model tuning, and deployment, this approach can be extended to real-world applications such as human–computer interaction, mental health monitoring, and user experience analysis.


# 👤 Author

Toahir Hussain

Machine Learning & Data Analytics Enthusiast
