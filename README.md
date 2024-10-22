# Handwritten Digit Recognition Project

This project demonstrates a machine learning model for recognizing handwritten digits from the **MNIST dataset**. It includes both a backend for model training and a **front-end interface** for live digit predictions.

## Project Overview
The goal of this project is to develop an effective digit recognition model using **Convolutional Neural Networks (CNNs)** and provide a user-friendly front-end for real-time predictions.

## Data Preprocessing and Augmentation
- Used the **MNIST dataset** containing 60,000 training images and 10,000 test images of handwritten digits (0-9).
- Applied **data augmentation techniques** like rotation, zoom, and shifting to improve generalization and prevent overfitting.

## Model Implementation
- Built a **CNN** using **TensorFlow** and **Keras**.
- Implemented:
  - **ReLU activation functions**
  - **Dropout layers** for regularization
  - **Batch normalization** for faster convergence
- Used **softmax activation** in the final layer for multi-class classification.
- Achieved **98%+ accuracy** on the test dataset.

## Front-End Interface
- Developed a **web-based front-end** using **HTML, CSS, and JavaScript**.
- Implemented a **canvas feature** where users can draw a digit, and the model predicts it in real time.
- Connected the front-end to the backend using **Flask** (or any other framework used) to provide instant predictions.

## Model Evaluation
- Evaluated performance using:
  - **Accuracy**
  - **Confusion matrix**
  - **Cross-entropy loss**

## Technologies Used
- **Backend:** TensorFlow, Keras, Flask (or relevant framework)  
- **Frontend:** HTML, CSS, JavaScript  
- **Other Libraries:** Pandas, NumPy, Matplotlib  

---

## How to Run the Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/HandwritingRecognitonProject.git
   cd DigitRecognitionProject
