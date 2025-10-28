# Digit Recognizer

A Python machine learning project for hand-written digit recognition using the MNIST dataset with scikit-learn and TensorFlow/Keras.

## Overview

This project implements a Convolutional Neural Network (CNN) and traditional machine learning approaches to recognize hand-written digits (0-9) from the MNIST dataset. The MNIST dataset contains 70,000 grayscale images of hand-written digits (60,000 for training and 10,000 for testing).

## Features

- Load and preprocess MNIST dataset
- Implement CNN model using TensorFlow/Keras
- Train and evaluate the model
- Make predictions on new digit images
- Visualize results and accuracy metrics

## Requirements

Install the required dependencies:

```bash
pip install numpy tensorflow scikit-learn matplotlib
```

## Usage

### Running the Digit Recognizer

```bash
python digit_recognizer.py
```

The script will:
1. Load the MNIST dataset
2. Preprocess and normalize the data
3. Build a CNN model
4. Train the model on the training data
5. Evaluate the model on test data
6. Display sample predictions with visualizations

## Model Architecture

The CNN model includes:
- Input layer (28x28 grayscale images)
- Convolutional layers with ReLU activation
- Max pooling layers
- Dropout for regularization
- Dense layers
- Output layer with softmax activation (10 classes)

## Results

The model typically achieves:
- Training accuracy: ~99%
- Test accuracy: ~98%+

## Project Structure

```
digit-recognizer/
├── README.md              # Project documentation
└── digit_recognizer.py    # Main Python script
```

## License

This project is open source and available for educational purposes.

## Acknowledgments

- MNIST dataset: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- TensorFlow/Keras documentation
- scikit-learn community
