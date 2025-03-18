#!/usr/bin/env python
"""
Simple MNIST example using Keras Leaderboard.

This example trains and compares several models on the MNIST dataset
and tracks their performance using the Keras Leaderboard.
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from keras_leaderboard import KerasLeaderboard


def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape and normalize
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    
    # One-hot encode targets
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)


def create_simple_model():
    """Create a simple CNN model for MNIST."""
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model


def create_deeper_model():
    """Create a deeper CNN model for MNIST."""
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model


def main():
    """Main function to run the example."""
    # Create output directory
    output_dir = "mnist_leaderboard_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the leaderboard
    leaderboard = KerasLeaderboard(output_dir=output_dir)
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Train simple model
    print("Training simple model...")
    simple_model = create_simple_model()
    simple_history = simple_model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=5,
        validation_split=0.1,
        verbose=1
    )
    
    # Add simple model to leaderboard
    leaderboard.add_model(simple_model, "simple_cnn", history=simple_history)
    
    # Train deeper model
    print("Training deeper model...")
    deeper_model = create_deeper_model()
    deeper_history = deeper_model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=5,
        validation_split=0.1,
        verbose=1
    )
    
    # Add deeper model to leaderboard
    leaderboard.add_model(deeper_model, "deeper_cnn", history=deeper_history)
    
    # Display the leaderboard
    leaderboard.display()
    
    # Export the leaderboard to CSV
    leaderboard.export_csv(os.path.join(output_dir, "mnist_leaderboard.csv"))
    print(f"Leaderboard saved to {output_dir}/mnist_leaderboard.csv")


if __name__ == "__main__":
    main() 