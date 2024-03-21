# Banknote Recognition with Machine Learning

Banknote Recognition with Machine Learning is a project aimed at utilizing machine learning techniques to recognize and classify banknotes based on their images. The project employs various machine learning algorithms to train models that can accurately identify the denominations of banknotes.

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
  - [Training and Evaluation](#training-and-evaluation)
  - [Detecting New Banknotes](#detecting-new-banknotes)
- [Example Usage](#example-usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Banknote recognition plays a crucial role in various applications, including automated teller machines (ATMs), currency counting machines, and retail transactions. This project aims to provide a solution to automate the recognition process using machine learning algorithms.

The workflow involves preprocessing banknote images, training machine learning models, and then using these models to predict the denominations of new banknotes. The project offers flexibility in choosing between different machine learning algorithms such as Support Vector Machines (SVM), k-Nearest Neighbors (KNN), and Perceptron.

## Technologies Used

- **Python:** The project is developed using Python programming language due to its simplicity and extensive libraries for machine learning.
- **OpenCV:** OpenCV is used for image processing tasks such as resizing, grayscale conversion, and feature extraction.
- **NumPy:** NumPy is utilized for numerical computations and handling multidimensional arrays, which are prevalent in image data.
- **Scikit-learn:** Scikit-learn library provides various machine learning algorithms and tools for model training, evaluation, and preprocessing.
- **Matplotlib:** Matplotlib is used for visualizing data and results, facilitating easy interpretation of model performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/banknote-recognition.git
   cd banknote-recognition
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training and Evaluation

1. Prepare the dataset: Place banknote images in the `data` directory, organized into subdirectories based on their denominations (e.g., `data/5`, `data/10`, etc.).

2. Train and evaluate the models:
   ```bash
   python train_and_evaluate.py
   ```

### Detecting New Banknotes

1. Choose a model (SVM, KNN, or Perceptron).

2. Provide the path of the new banknote image to be recognized.

3. Execute the detection script:
   ```bash
   python detect_banknote.py
   ```

## Example Usage

1. Select the SVM model.

2. Enter the path of the new banknote image: `path/to/new/banknote/image.jpg`.

3. View the predicted denomination of the banknote.

## Contributing

Contributions to the project are welcome! If you have any suggestions, bug reports, or feature requests, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

