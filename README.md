# Neural Network Optimizer Comparison

This project implements a neural network from scratch for a multiclass classification task. It compares the performance of various optimization algorithms, including Gradient Descent, on a generated dataset.

## Features

- Neural network architecture:
  - Input layer: 4 neurons (features)
  - First hidden layer: 3 neurons
  - Second hidden layer: 4 neurons
  - Output layer: 3 neurons (classes)
- Activation functions:
  - **ReLU**: Used in hidden layers
  - **Softmax**: Used in the output layer for multiclass classification
- Loss function:
  - **Categorical Cross-Entropy**
- Optimizers:
  - Gradient Descent
  - (Extendable to Momentum GD, NAG, AdaGrad, RMSProp, Adam, etc.)
- Performance metrics:
  - Accuracy

## Getting Started

### Prerequisites

- Python 3.6+
- NumPy
- scikit-learn

Install the dependencies using pip:
```bash
pip install numpy scikit-learn
```

### Running the Code

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd nn-optimizer-comparison
   ```
2. Run the Python script:
   ```bash
   python neural_network.py
   ```

### Example Dataset

- A synthetic dataset is generated using `make_classification` from scikit-learn.
- Features: 4
- Classes: 3

## Neural Network Workflow

1. **Data Preparation**:
   - Generate synthetic dataset.
   - Convert labels to one-hot encoding.
   - Split data into training and testing sets.
2. **Model Initialization**:
   - Randomly initialize weights and biases.
3. **Forward Propagation**:
   - Compute layer outputs using ReLU and Softmax activations.
4. **Backward Propagation**:
   - Calculate gradients of the loss function with respect to weights and biases.
5. **Optimization**:
   - Update weights and biases using the specified optimizer.
6. **Evaluation**:
   - Measure loss and accuracy after each epoch.

## Code Overview

- **`neural_network.py`**:
  - Implements the neural network and training process.
  - Includes activation functions, loss computation, and backpropagation.
  - Supports adding custom optimizers.

## Future Enhancements

- Implement additional optimizers (Momentum GD, NAG, AdaGrad, RMSProp, Adam).
- Add visualization of loss and accuracy trends.
- Extend to real-world datasets.

