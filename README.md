# Cats-vs-Dogs-CNN-Image-Classifier-Deep-Learning-Essentials-
give an image → the model predicts - ex : cat and dog

“I built a Cat vs Dog image classifier using a Convolutional Neural Network with three convolution–pooling blocks. The model extracts spatial features using filters, reduces dimensions with max pooling, and classifies using dense layers. I used ImageDataGenerator for preprocessing and achieved around 85–90% accuracy after 5 epochs.”


This demonstrates:

Convolution layers

Max Pooling

Activation functions

Softmax

Training/validation split

Overfitting control


# Cats vs Dogs Image Classification using CNN

This project demonstrates a simple Convolutional Neural Network (CNN) for binary image
classification using the Kaggle Cats vs Dogs dataset.

## Model Architecture
- Conv2D → MaxPool
- Conv2D → MaxPool
- Conv2D → MaxPool
- Flatten
- Dense (ReLU)
- Dense (Sigmoid)

## Accuracy
~85–90% after 5 epochs.

## Dataset
Kaggle: Dogs vs Cats  
(Not uploaded to GitHub due to size — add to .gitignore)

📁 GitHub Folder Structure

cnn-cat-dog-classifier/ 

│── model.ipynb

│── saved_model/

│── README.md

│── .gitignore

