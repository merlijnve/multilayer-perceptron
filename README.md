# Multilayer Perceptron

A from-scratch implementation of a multilayer perceptron (MLP) neural network in Python for binary classification tasks. This project demonstrates the fundamentals of artificial neural networks without relying on high-level frameworks like TensorFlow or PyTorch.

## 📋 Overview

This implementation includes:
- **Custom neural network architecture** with configurable layers
- **Multiple activation functions** (Sigmoid, ReLU, Softmax)
- **Backpropagation** algorithm for training
- **Early stopping** to prevent overfitting
- **Data normalization** layer
- **Visualization tools** for dataset exploration
- **Binary classification** on the Wisconsin Breast Cancer dataset

## 🚀 Features

- ✨ Built from scratch using only NumPy for numerical operations
- 📊 Training and validation loss/accuracy tracking
- 🎯 Early stopping mechanism for optimal model selection
- 🔄 Configurable network architecture (layer sizes and depths)
- 📈 Real-time plotting of training metrics
- 💾 Model serialization for saving and loading trained networks
- 🧪 Dataset splitting utilities
- 📉 Multiple loss functions (cross-entropy, binary cross-entropy)

## 📦 Installation

### Prerequisites
- Python 3.x
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/merlijnve/multilayer-perceptron.git
cd multilayer-perceptron
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy` - Numerical computations and array operations
- `matplotlib` - Plotting training curves and visualizations
- `pandas` - Dataset loading and manipulation
- `seaborn` - Statistical data visualization

## 📖 Usage

### 1. Split the Dataset

Split the breast cancer dataset into training and test sets:

```bash
python -m mlp.split data/breast_cancer_data.csv 20
```

This creates two files:
- `data/breast_cancer_data.csv_train` (80% of data)
- `data/breast_cancer_data.csv_test` (20% of data)

Arguments:
- First argument: path to the dataset CSV file
- Second argument: percentage of data to use for testing (e.g., 20 for 20%)

### 2. Train the Model

Train a neural network on the training dataset:

```bash
python -m mlp.train
```

The training script will:
- Load the training data from `data/breast_cancer_data.csv_train`
- Create a neural network with:
  - Normalization layer
  - Dense layer: 30 inputs → 64 neurons (Sigmoid)
  - Dense layer: 64 → 64 neurons (Sigmoid)
  - Output layer: 64 → 2 outputs (Softmax)
- Train for up to 400 epochs with early stopping
- Display training progress with loss and accuracy plots
- Save the best model to `models/model.pkl`

The default architecture can be modified in `mlp/train.py`.

### 3. Make Predictions

Use a trained model to make predictions on the test dataset:

```bash
python -m mlp.predict data/breast_cancer_data.csv_test models/model.pkl
```

Output example:
```
Correct: 114/114
Accuracy: 1.0
Binary cross entropy loss: 0.0234
```

Arguments:
- First argument: path to the test dataset
- Second argument: path to the saved model file

### 4. Visualize the Dataset (Optional)

Generate pairplot visualizations of the dataset features:

```bash
python -m mlp.visualize_dataset
```

This creates two visualization files in the `images/` directory:
- `pairplot.png` - Original feature distributions
- `pairplot_normalized.png` - Normalized feature distributions

⚠️ **Note**: This operation can take several minutes due to the high number of features (30).

## 📊 Dataset

The project uses the **Wisconsin Breast Cancer Dataset**, which contains features computed from digitized images of fine needle aspirate (FNA) of breast masses.

### Dataset Details
- **Instances**: 569 samples
- **Features**: 30 real-valued features (plus ID and diagnosis)
- **Classes**: 
  - Malignant (M): 212 samples
  - Benign (B): 357 samples
- **Task**: Binary classification (cancer diagnosis)

### Features
Ten characteristics are computed for each cell nucleus:
1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave points
9. Symmetry
10. Fractal dimension

For each characteristic, three values are provided:
- Mean
- Standard error
- Worst (mean of three largest values)

See `dataset_explanation.md` for detailed information about the dataset.

## 🏗️ Project Structure

```
multilayer-perceptron/
├── mlp/
│   ├── __init__.py
│   ├── NeuralNetwork.py         # Main neural network class
│   ├── DenseLayer.py            # Fully connected layer implementation
│   ├── NormalizationLayer.py    # Data normalization layer
│   ├── activation_functions.py  # Sigmoid, ReLU, Softmax activations
│   ├── support_functions.py     # Utility functions (loss, normalization)
│   ├── train.py                 # Training script
│   ├── predict.py               # Prediction script
│   ├── split.py                 # Dataset splitting utility
│   └── visualize_dataset.py     # Dataset visualization tool
├── data/
│   └── breast_cancer_data.csv   # Wisconsin Breast Cancer dataset
├── images/                      # Generated visualization plots
├── models/                      # Saved trained models
├── requirements.txt             # Python dependencies
├── dataset_explanation.md       # Detailed dataset documentation
└── README.md                    # This file
```

## 🧠 Architecture Details

### Neural Network Components

**NeuralNetwork Class** (`NeuralNetwork.py`)
- Manages the entire network architecture
- Implements forward and backward propagation
- Handles training loop with validation
- Includes early stopping mechanism
- Plots training metrics in real-time

**DenseLayer Class** (`DenseLayer.py`)
- Fully connected layer with configurable size
- Weight initialization using uniform distribution
- Bias terms for each neuron
- Forward pass: activation(Wx + b)
- Backward pass: gradient computation and weight updates

**NormalizationLayer Class** (`NormalizationLayer.py`)
- Preprocessing layer for input normalization
- Computes mean and standard deviation
- Normalizes features to zero mean and unit variance

**Activation Functions**
- **Sigmoid**: Smooth gradient, outputs in (0, 1)
- **ReLU**: Faster training, prevents vanishing gradients
- **Softmax**: Multi-class probability distribution (used for output layer)

### Training Process

1. **Forward Propagation**: Input passes through each layer
2. **Loss Calculation**: Cross-entropy loss between predictions and targets
3. **Backward Propagation**: Gradients computed from output to input
4. **Weight Updates**: Gradient descent updates network parameters
5. **Validation**: Model evaluated on validation set
6. **Early Stopping**: Training stops if validation loss doesn't improve

## 🎓 Learning Objectives

This project demonstrates understanding of:
- Neural network fundamentals (forward/backward propagation)
- Gradient descent optimization
- Activation functions and their derivatives
- Regularization techniques (early stopping)
- Data preprocessing and normalization
- Model evaluation metrics
- Python OOP principles

## 📝 Example: Custom Network Architecture

You can create custom architectures by modifying `train.py`:

```python
from mlp.NeuralNetwork import NeuralNetwork
from mlp.DenseLayer import DenseLayer
from mlp.NormalizationLayer import NormalizationLayer
from mlp.activation_functions import Sigmoid, ReLU, Softmax

# Create a custom network
network = NeuralNetwork([
    NormalizationLayer(),
    DenseLayer(30, 128, ReLU(), learning_rate=0.01),
    DenseLayer(128, 64, ReLU(), learning_rate=0.01),
    DenseLayer(64, 32, Sigmoid(), learning_rate=0.01),
    DenseLayer(32, 2, Softmax(), learning_rate=0.1)
], early_stopping_n_epochs=15)

# Train the network
network.fit(X_train, y_train, epochs=500, plotting=True, batch_size=32)
```

## 📈 Performance

The default network architecture typically achieves:
- **Training Accuracy**: ~98-99%
- **Validation Accuracy**: ~97-99%
- **Test Accuracy**: ~97-100%

Results may vary based on:
- Random weight initialization
- Train/test split
- Hyperparameters (learning rates, layer sizes)
- Number of training epochs

## 🔧 Troubleshooting

**Issue**: Training loss is not decreasing
- Try adjusting learning rates (increase or decrease)
- Check if gradients are vanishing (use ReLU instead of Sigmoid)
- Ensure data is properly normalized

**Issue**: Overfitting (training accuracy >> validation accuracy)
- Reduce network complexity (fewer layers/neurons)
- Decrease early stopping patience
- Add more training data

**Issue**: Import errors
- Make sure to run scripts as modules: `python -m mlp.train`
- Verify all dependencies are installed: `pip install -r requirements.txt`

## 🤝 Contributing

This is an educational project demonstrating neural network fundamentals. Contributions, suggestions, and improvements are welcome!

## 📄 License

This project is available for educational purposes.

## 🙏 Acknowledgments

- Dataset: [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Inspired by foundational neural network research and educational materials

## 📚 Further Reading

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
