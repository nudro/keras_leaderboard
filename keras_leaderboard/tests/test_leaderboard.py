"""
Unit tests for the Keras Leaderboard module.
"""

import os
import shutil
import unittest
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_leaderboard import KerasLeaderboard


class TestKerasLeaderboard(unittest.TestCase):
    """Test cases for the KerasLeaderboard class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create a simple test model
        self.model = keras.Sequential([
            keras.layers.Dense(10, activation='relu', input_shape=(20,)),
            keras.layers.Dense(5, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create dummy data and train model
        x = np.random.random((100, 20))
        y = keras.utils.to_categorical(np.random.randint(5, size=(100, 1)), 5)
        
        self.history = self.model.fit(
            x, y,
            epochs=2,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary test directory
        shutil.rmtree(self.test_dir)
    
    def test_create_leaderboard(self):
        """Test creating a leaderboard instance."""
        leaderboard = KerasLeaderboard(output_dir=self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))
        
    def test_add_model(self):
        """Test adding a model to the leaderboard."""
        leaderboard = KerasLeaderboard(output_dir=self.test_dir)
        leaderboard.add_model(self.model, "test_model", history=self.history)
        
        # Check if model directory was created
        model_dirs = [d for d in os.listdir(self.test_dir) 
                       if os.path.isdir(os.path.join(self.test_dir, d)) and 
                       d.startswith("test_model_")]
        
        self.assertTrue(len(model_dirs) > 0)
    
    def test_export_csv(self):
        """Test exporting the leaderboard to CSV."""
        leaderboard = KerasLeaderboard(output_dir=self.test_dir)
        leaderboard.add_model(self.model, "test_model", history=self.history)
        
        csv_path = os.path.join(self.test_dir, "test_leaderboard.csv")
        leaderboard.export_csv(csv_path)
        
        self.assertTrue(os.path.exists(csv_path))


if __name__ == "__main__":
    unittest.main() 