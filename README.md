

# Credit Card Fraud Detection Using Convolutional Neural Networks (CNN)

This repository contains a machine learning project for detecting fraudulent credit card transactions using Convolutional Neural Networks (CNNs). The goal of the project is to predict whether a credit card transaction is legitimate or fraudulent based on transaction features.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Credit card fraud is a serious issue in the financial sector, causing millions of dollars in losses annually. This project aims to apply Convolutional Neural Networks (CNN), which are typically used for image processing, to time-series transaction data to detect fraudulent transactions.

By utilizing CNNs, we attempt to capture spatial patterns in transaction data that may help improve fraud detection accuracy.

## Dataset

The dataset used for this project is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), available on Kaggle. It contains anonymized credit card transaction data, with features like:

- `Time`: The number of seconds elapsed between this transaction and the first transaction in the dataset.
- `V1, V2, V3,...V28`: 28 anonymized features (resulting from PCA transformation).
- `Amount`: The amount of the transaction.
- `Class`: Whether the transaction is fraudulent (1) or legitimate (0).

### Important Notes:
- The dataset is highly imbalanced, with fraudulent transactions making up a very small portion of the data.
- The original dataset has been preprocessed, so you only need to focus on feature engineering and model development.

## Technologies Used

- **Python**: Programming language used for data preprocessing, model building, and evaluation.
- **TensorFlow**: Deep learning framework used to build and train the CNN model.
- **Keras**: High-level neural networks API, running on top of TensorFlow.
- **NumPy**: Used for numerical computing and data manipulation.
- **Pandas**: Used for data analysis and manipulation.
- **Matplotlib**: For visualizing data and model performance.
- **Scikit-learn**: For data preprocessing and model evaluation metrics.

## Model Architecture

The model is a simple Convolutional Neural Network (CNN) architecture designed to identify patterns in the transaction features. The architecture is as follows:

1. **Input Layer**: Receives the input transaction data (preprocessed).
2. **Convolutional Layer**: Extracts local patterns in the data using convolution operations.
3. **Max Pooling Layer**: Reduces dimensionality and retains important features.
4. **Flatten Layer**: Converts the 2D matrix data into a 1D vector.
5. **Fully Connected Layer**: A dense layer that processes the flattened data.
6. **Output Layer**: Produces a binary classification output (fraudulent or legitimate).

The model is trained using binary cross-entropy loss function and evaluated using accuracy, precision, recall, and F1-score to assess performance, particularly with respect to imbalanced data.

## Installation

To run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection-cnn.git
   cd credit-card-fraud-detection-cnn
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset:
   - Go to [this link](https://www.kaggle.com/mlg-ulb/creditcardfraud) and download the dataset. Place it in the `data/` folder.

## How to Use

1. Preprocess the data by running the preprocessing script:
   ```bash
   python preprocess.py
   ```

2. Train the CNN model:
   ```bash
   python train_model.py
   ```

3. After training, evaluate the model:
   ```bash
   python evaluate_model.py
   ```

4. For making predictions on new data:
   ```bash
   python predict.py --input_path data/new_transactions.csv
   ```

## Results

The model is evaluated based on the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Due to the highly imbalanced nature of the dataset, metrics such as precision, recall, and F1-score are more useful for evaluating the model's performance in detecting fraudulent transactions.

### Sample Model Performance:
- **Accuracy**: 98.5%
- **Precision**: 0.92
- **Recall**: 0.83
- **F1-Score**: 0.87

Note: The model's high accuracy may be misleading due to class imbalance, so F1-Score and Recall are the primary evaluation metrics.

## Contributing

Contributions are welcome! If you'd like to improve this project or add new features, feel free to fork the repository and create a pull request. Before contributing, please ensure that your code follows the existing coding style and passes all tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

### Additional Notes:
- **requirements.txt**: This file should include all the dependencies needed to run the project. For example:
  ```
  tensorflow==2.8.0
  numpy==1.21.0
  pandas==1.3.0
  scikit-learn==0.24.2
  matplotlib==3.4.3
  ```

This `README.md` serves as an introduction and instruction manual for anyone who wants to understand, run, or contribute to the project. Make sure to adapt the text as needed, especially the "Results" section once you have actual results from running the model.
