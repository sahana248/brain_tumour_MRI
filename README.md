# Comparative Analysis of deep learning models for MRI- Based brain tumour Classification.
## Research Question
Compare the effectiveness of custom deep learning models (GCN and GAN) with pretrained models (VGG19 and ResNet50) for brain tumor classification.

## Project Objectives
In this project, we aim to develop and evaluate deep learning models, including Graph Convolutional Networks (GCN) and Generative Adversarial Networks (GAN), for brain tumour classification using MRI images. Our approach involves training and fine-tuning these custom models on a labelled dataset of brain MRI scans. We will then compare their performance with pretrained models such as VGG19 and ResNet50. Evaluation metrics, including accuracy, precision, recall, F1-score, and ROC-AUC, will guide us in determining the effectiveness of each model.

## Summary and Background
This study explores the effectiveness of deep learning techniques on MRI images for the accurate classification of brain tumors. Brain tumors present major diagnostic challenges that require fast and accurate classification methods. Traditional manual examinations conducted by radiologists are labor-intensive and error-prone, highlighting the need for automated solutions.

This project will specifically compare pretrained models, such as VGG19 and ResNet50, with custom models, including Graph Convolutional Networks (GCN) and Generative Adversarial Networks (GAN). Pretrained models benefit from extensive training on large datasets and sophisticated architectures, whereas custom models offer advantages like enhanced spatial relationship analysis and the generation of synthetic data to improve training outcomes. By evaluating these models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC, the research aims to identify the most effective model for accurate and efficient brain tumor classification in medical imaging.

## Technologies Used
1. Python
2. TensorFlow
3. Keras
4. PyTorch
5. Scikit-learn
6. NumPy
7. Pandas
8. Matplotlib

## Installation and Setup
### Prerequisites
Python 3.8 or higher
GPU (recommended for training deep learning models)
### Dependencies
Install the required dependencies using pip:
' pip install -r requirements.txt '
### Dataset
Download the MRI brain tumor dataset.  You can find the dataset here.
### Running the Project
1. Preprocess the data:
  python preprocess.py
2. Train the models:
  python train.py --model gcn
  python train.py --model gan
  python train.py --model vgg19
  python train.py --model resnet50
3. Evaluate the models:
  python evaluate.py --model gcn
  python evaluate.py --model gan
  python evaluate.py --model vgg19
  python evaluate.py --model resnet50
### Configuration
Adjust the configuration settings in 'config.json' to customize parameters such as learning rate, batch size, and number of epochs.
## Contributors
Sahana Muralidaran (sm22adg@herts.ac.uk)
### License
This project is licensed under the MIT License - see the LICENSE file for details.
