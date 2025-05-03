# Multimodal-Ensemble-Deep-Learning-Skin-Cancer-Classification


This project leverages machine learning and deep learning techniques to detect skin cancer using dermoscopic images and clinical metadata. The model uses a multimodal ensemble approach that combines multiple image classification models and metadata for accurate skin cancer classification.

## Overview

Skin cancer is one of the most common types of cancer worldwide. Early detection is crucial to improve survival rates, and this project aims to build a robust model for classifying skin cancer using advanced techniques like convolutional neural networks (CNNs) and machine learning.

The project uses the **PAD-UFES-20** dataset, which contains dermoscopic images of various skin lesions, along with metadata about the patients. The goal is to classify images into categories like "Skin Cancer" or "Skin Disease."

## Dataset

The dataset used in this project is **PAD-UFES-20**. It contains:

- Dermoscopic images of skin lesions.
- Metadata such as patient age, lesion details, and diagnostic information.

The dataset has been preprocessed to ensure the quality and compatibility of images and metadata for model training.

## Models

The project uses an ensemble of different models, including:

1. **DenseNet**: A deep convolutional neural network model that performs well for image classification tasks.
2. **InceptionNet**: Another powerful model used to classify images based on learned features.
3. **EfficientNet**: A model that balances accuracy and computational efficiency.
4. **MLP (Multilayer Perceptron)**: A meta-learner that processes metadata to improve predictions.
5. **Stacking Ensemble**: A combination of the above models, where the outputs are used for a final prediction.


## Technologies Used

- **Python**: Main programming language for implementing the models.
- **TensorFlow/Keras**: For building and training deep learning models.
- **Scikit-learn**: For implementing machine learning algorithms and evaluation metrics.
- **OpenCV**: For image preprocessing and augmentation.
- **Pandas/Numpy**: For data manipulation and preprocessing.
