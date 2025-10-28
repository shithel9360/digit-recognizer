#!/usr/bin/env python3
"""
Digit Recognizer - MNIST Hand-written Digit Recognition

This script implements a Convolutional Neural Network (CNN) using TensorFlow/Keras
to recognize hand-written digits from the MNIST dataset.

Author: Digit Recognizer Project
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DigitRecognizer:
    def __init__(self):
        """Initialize the Digit Recognizer."""
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.history = None
        
    def load_data(self):
        """Load and preprocess the MNIST dataset."""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values to range [0, 1]
        self.x_train = x_train.astype('float32') / 255.0
        self.x_test = x_test.astype('float32') / 255.0
        
        # Reshape data to add channel dimension (28, 28, 1)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
        
        # Convert labels to categorical one-hot encoding
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)
        
        print(f"Training data shape: {self.x_train.shape}")
        print(f"Training labels shape: {self.y_train.shape}")
        print(f"Test data shape: {self.x_test.shape}")
        print(f"Test labels shape: {self.y_test.shape}")
        
    def build_model(self):
        """Build the CNN model architecture."""
        print("Building CNN model...")
        
        self.model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
        ])
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        self.model.summary()
        
    def train_model(self, epochs=10, batch_size=128):
        """Train the CNN model."""
        print(f"Training model for {epochs} epochs...")
        
        # Train the model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_test, self.y_test),
            verbose=1
        )
        
        print("Training completed!")
        
    def evaluate_model(self):
        """Evaluate the trained model on test data."""
        print("Evaluating model on test data...")
        
        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Get predictions
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes))
        
        return test_accuracy, y_pred_classes, y_true
        
    def plot_training_history(self):
        """Plot training and validation accuracy/loss."""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot training & validation accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
    def predict_samples(self, num_samples=10):
        """Make predictions on random test samples and visualize results."""
        if self.model is None:
            print("Model not trained. Train the model first.")
            return
            
        # Select random samples from test set
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        sample_images = self.x_test[indices]
        sample_labels = np.argmax(self.y_test[indices], axis=1)
        
        # Make predictions
        predictions = self.model.predict(sample_images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Plot results
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            # Display image
            axes[i].imshow(sample_images[i].reshape(28, 28), cmap='gray')
            axes[i].set_title(f'True: {sample_labels[i]}, Pred: {predicted_labels[i]}')
            axes[i].axis('off')
            
            # Color title based on correctness
            if sample_labels[i] == predicted_labels[i]:
                axes[i].set_title(f'True: {sample_labels[i]}, Pred: {predicted_labels[i]}', 
                                color='green')
            else:
                axes[i].set_title(f'True: {sample_labels[i]}, Pred: {predicted_labels[i]}', 
                                color='red')
                
        plt.tight_layout()
        plt.show()
        
        return sample_images, sample_labels, predicted_labels

def main():
    """Main function to run the digit recognition pipeline."""
    print("=" * 50)
    print("MNIST Digit Recognition with CNN")
    print("=" * 50)
    
    # Initialize the digit recognizer
    recognizer = DigitRecognizer()
    
    # Load and preprocess data
    recognizer.load_data()
    
    # Build the model
    recognizer.build_model()
    
    # Train the model
    recognizer.train_model(epochs=5, batch_size=128)  # Reduced epochs for demo
    
    # Evaluate the model
    test_accuracy, y_pred, y_true = recognizer.evaluate_model()
    
    # Plot training history
    recognizer.plot_training_history()
    
    # Plot confusion matrix
    recognizer.plot_confusion_matrix(y_true, y_pred)
    
    # Make predictions on sample images
    print("\nMaking predictions on random test samples...")
    recognizer.predict_samples(num_samples=10)
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    print("Digit recognition completed!")

if __name__ == "__main__":
    main()
