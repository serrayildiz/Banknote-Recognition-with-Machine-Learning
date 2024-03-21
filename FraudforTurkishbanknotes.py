import numpy as np
import cv2
from sklearn import svm, neighbors, linear_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# Function to preprocess the banknote images
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return None
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'.")
        return None
    resized_image = cv2.resize(image, (100, 100))
    flattened_image = resized_image.flatten()
    return flattened_image

# Function to load dataset
def load_dataset(notes):
    X = []
    y = []
    for note in notes:
        for i in range(1, 16):
            img_path = f'data/{note}/{i}.jpg'  # Assuming images are stored in folders named after the notes
            preprocessed_img = preprocess_image(img_path)
            if preprocessed_img is not None:
                X.append(preprocessed_img)
                y.append(note)
    return np.array(X), np.array(y)

# Function to train SVM model
def train_svm(X_train, y_train):
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    return clf

# Function to train KNN model
def train_knn(X_train, y_train):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    return clf

# Function to train Perceptron model
def train_perceptron(X_train, y_train):
    clf = linear_model.Perceptron()
    clf.fit(X_train, y_train)
    return clf

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Function for user to select model
def select_model(model_name):
    if model_name.lower() == 'svm':
        return 'SVM'
    elif model_name.lower() == 'knn':
        return 'KNN'
    elif model_name.lower() == 'perceptron':
        return 'Perceptron'
    else:
        print("Invalid model selection.")
        return None

# Load dataset
notes = ['5', '10', '20', '50', '100', '200']
X, y = load_dataset(notes)

# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
svm_model = train_svm(X_train, y_train)
knn_model = train_knn(X_train, y_train)
perceptron_model = train_perceptron(X_train, y_train)

# Evaluate models
svm_accuracy = evaluate_model(svm_model, X_test, y_test)
knn_accuracy = evaluate_model(knn_model, X_test, y_test)
perceptron_accuracy = evaluate_model(perceptron_model, X_test, y_test)

print(f"SVM Accuracy: {svm_accuracy}")
print(f"KNN Accuracy: {knn_accuracy}")
print(f"Perceptron Accuracy: {perceptron_accuracy}")

# User interface for selecting model and detecting new banknote image
def detect_banknote(model, image_path):
    preprocessed_img = preprocess_image(image_path)
    if preprocessed_img is not None:
        prediction = model.predict([preprocessed_img])[0]
        return prediction
    else:
        return None

# Example of using the user interface
selected_model = input("Select a model (SVM, KNN, or Perceptron): ")
selected_model = select_model(selected_model)

if selected_model:
    new_banknote_image_path = input("Enter the path of the new banknote image: ")
    prediction = detect_banknote(svm_model, new_banknote_image_path)
    if prediction is not None:
        print(f"Predicted denomination: {prediction}")
