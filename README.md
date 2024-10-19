# TensorFlow for Deep Learning Introudction Course

This repository contains the materials and code for the **TensorFlow for Deep Learning** course by Google, hosted on Udacity. The course provides a practical introduction to using TensorFlow for building deep learning models and covers a variety of topics, including machine learning fundamentals, convolutional neural networks (CNNs), and more.

**Course link**: [Udacity TensorFlow for Deep Learning](https://www.udacity.com/enrollment/ud187)

## Table of Contents
- [Course Overview](#course-overview)
- [Folders Description](#folders-description)
  - [0- TensorFlow Cuda Support](#0--tensorflow-cuda-support)
  - [1- Welcome to the Course](#1--welcome-to-the-course)
  - [2- Introduction to Machine Learning](#2--introduction-to-machine-learning)
  - [3- Fashion MNIST](#3--fashion-mnist)
  - [4- Introduction to CNNs](#4--introduction-to-cnns)
  - [5- CNNs Continued](#5--cnns-continued)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Course Overview

The purpose of this project is to demonstrate how to use TensorFlow for training and deploying machine learning models. Specifically, it focuses on building deep learning models for image classification using convolutional neural networks (CNNs). The project contains a series of notebooks that cover TensorFlow setup, introduction to machine learning, and more advanced topics like CNNs.

## Folders Description

### 0- TensorFlow Cuda Support
   - **Description**: This folder contains resources and setup instructions for enabling CUDA support on TensorFlow. CUDA is NVIDIA's platform for parallel computing that allows TensorFlow to run computations on GPUs, significantly accelerating model training.

### 1- Welcome to the Course
   - **Description**: The introductory folder for the course. It likely contains welcome materials and instructions on how to navigate the course and set up your environment for deep learning with TensorFlow.

### 2- Introduction to Machine Learning
   - **Description**: This folder introduces the basic concepts of machine learning, including supervised learning, unsupervised learning, and key algorithms like linear regression, logistic regression, and more.

### 3- Fashion MNIST
   - **Description**: This folder contains the code and materials for the Fashion MNIST dataset project. The Fashion MNIST dataset is a collection of grayscale images of fashion items (e.g., shoes, shirts) used for image classification tasks. You'll learn how to build a TensorFlow model that classifies these images into different categories.

### 4- Introduction to CNNs
   - **Description**: This folder provides an introduction to convolutional neural networks (CNNs), a type of deep learning model particularly effective for image recognition tasks. You'll learn the theory behind CNNs and how to implement them in TensorFlow.

### 5- CNNs Continued
   - **Description**: This folder continues the exploration of CNNs, diving deeper into advanced CNN architectures and techniques. You'll learn about techniques like pooling, dropout, and data augmentation to improve model performance.

## Setup and Installation

### Prerequisites
To run the notebooks and code in this repository, you'll need:
- Python 3.x installed.
- TensorFlow installed (with optional GPU support).
- Jupyter Notebook or JupyterLab installed (`pip install jupyterlab`).
- An NVIDIA GPU with CUDA installed (optional for GPU acceleration).

### Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/username/TensorFlow-For-Deep-Learning.git
    ```

2. Navigate to the project directory:
    ```bash
    cd TensorFlow-For-Deep-Learning
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. To use GPU acceleration, install CUDA and cuDNN from NVIDIA's website and configure TensorFlow to use the GPU.

## Usage

1. Launch JupyterLab or Jupyter Notebook:
    ```bash
    jupyter lab
    ```

2. Open any of the provided Jupyter notebooks in the folders:
    - In `0- TensorFlow Cuda Support`: Set up TensorFlow with CUDA.
    - In `1- Welcome to the Course`: Follow introductory materials and ensure your environment is correctly set up.
    - In `3- Fashion MNIST`: Train a CNN on the Fashion MNIST dataset.

3. Execute the cells in each notebook to train models, view outputs, and complete the course exercises.

## Contributing

Contributions are welcome! If you'd like to improve the code, add new exercises, or suggest enhancements, please submit a pull request.

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Commit your changes:
    ```bash
    git commit -m 'Add new feature'
    ```
4. Push the branch:
    ```bash
    git push origin feature-branch
    ```
5. Open a Pull Request.

## Acknowledgements

Special thanks to Google and Udacity for providing this amazing course on deep learning with TensorFlow. Also, thanks to the TensorFlow community for building such a powerful machine learning framework.
