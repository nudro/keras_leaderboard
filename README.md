# Keras Leaderboard

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)

A lightweight framework for tracking, comparing, and visualizing Keras model performance and metadata.

![Leaderboard Preview](https://github.com/nudro/keras_leaderboard/blob/master/leaderboard_fin.gif)

## 🌟 Features

- Track and compare multiple Keras models in a single dashboard
- Automatic extraction of model metadata (parameters, layers, configurations)
- Interactive visualization of training metrics
- Export comparison data to CSV
- Supports custom CNN architectures and pre-trained models
- Integrates with Jupyter notebooks via qgrid widgets

## 📋 Requirements

- Python 3.6+
- TensorFlow 2.x
- Keras
- Pandas
- Matplotlib
- Jupyter (for notebook examples)
- qgrid (for interactive widgets)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nudro/keras_leaderboard.git
cd keras_leaderboard

# Create a virtual environment
conda create -n leaderboard_env python=3.6
conda activate leaderboard_env

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the Kaggle art dataset:
- [Art Images: Drawings, Paintings, Sculpture, Engravings](https://www.kaggle.com/thedownhill/art-images-drawings-painting-sculpture-engraving)

Place the downloaded data in a `data` directory within the project root.

### Running the Example

```bash
# Run the main script to train and compare models
python main.py
```

Or open and run the Jupyter notebook:
```bash
jupyter notebook Example.ipynb
```

## 📊 Usage

### Basic Usage

```python
from keras_leaderboard import KerasLeaderboard

# Initialize leaderboard
leaderboard = KerasLeaderboard()

# Build and train models (examples provided in the repo)
# ...

# Display leaderboard
leaderboard.display()
```

### Tracking Custom Models

```python
from keras_leaderboard import KerasLeaderboard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten

# Initialize leaderboard
leaderboard = KerasLeaderboard(output_dir="my_models")

# Define your custom model
inputs = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)

# Compile and fit your model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
            
# Train model
# history = model.fit(...)

# Add model to leaderboard
leaderboard.add_model(model, "my_custom_model", history=history)

# Display and export leaderboard
leaderboard.display()
leaderboard.export_csv("my_leaderboard.csv")
```

## 📁 Project Structure

```
keras_leaderboard/
├── keras_leaderboard.py    # Main library code
├── main.py                 # Example script
├── run_leaderboard.py      # CLI tool for running the leaderboard
├── Example.ipynb           # Jupyter notebook example
├── requirements.txt        # Project dependencies
└── data/                   # Directory for datasets (you need to add this)
```

## 📊 Output

When running models, outputs will be stored in subdirectories with the format:
`model_name_month_day_hour_minute`

Each directory contains:
- Training and validation metrics
- Model architecture visualizations
- Performance charts
- Metadata CSV files

## 📝 Contributing

Contributions are welcome! Here are some areas that could use improvement:

- Better visualization of model configurations (possibly using GraphViz)
- Fix for VGG model layer extraction in `get_model_outputs()`
- Expanded model support beyond 'basic_cnn' and 'basic_vgg'
- Automated hyperparameter tuning integration
- Frontend improvements using Plotly
- Database backend for model storage
- PyTorch implementation

Please feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Catherine Ordun
- Michael Fagundo
- Josh Luxton
- Chao Wu

## 📚 Citation

If you use this project in your research, please cite:

```
@software{keras_leaderboard,
  author = {Ordun, Catherine and Fagundo, Michael and Luxton, Josh and Wu, Chao},
  title = {Keras Leaderboard: A Framework for Tracking Neural Network Metadata},
  url = {https://github.com/nudro/keras_leaderboard},
  year = {2022},
}
``` 