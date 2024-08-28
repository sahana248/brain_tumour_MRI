# Comparative Analysis of deep learning models for MRI- Based brain tumour Classification.
## Research Question
How does the performance of custom deep learning models (GCN and GAN) compare to pretrained models (VGG19 and ResNet50) for brain tumour classification?

## Project Objectives
In this project, we aim to develop and evaluate deep learning models, including Graph Convolutional Networks (GCN) and Generative Adversarial Networks (GAN), for brain tumour classification using MRI images. Our approach involves training and fine-tuning these custom models on a labelled dataset of brain MRI scans. We will then compare their performance with pretrained models such as VGG19 and ResNet50. Evaluation metrics, including accuracy, precision, recall, F1-score, and ROC-AUC, will guide us in determining the effectiveness of each model.

## Summary and Background
This research investigates the efficacy of deep learning techniques applied to magnetic resonance imaging for accurate diagnosis of brain tumours. The detection of brain tumours is crucially important but it is generally difficult as their recognition requires trained experts. Consequently, this necessitates machine-aided approaches since manual examination is time-consuming and fraught with mistakes. 

This project will specifically compare pretrained models, such as VGG19 and ResNet50, with custom models, including Graph Convolutional Networks (GCN) and Generative Adversarial Networks (GAN). Pretrained models benefit from extensive training on large datasets and sophisticated architectures, whereas custom models offer advantages like enhanced spatial relationship analysis and the generation of synthetic data to improve training outcomes. By evaluating these models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC, the research aims to identify the most effective model for accurate and efficient brain tumor classification in medical imaging.

## Dataset
The dataset used in this project is sourced from Kaggle and consists of 7022 MRI images categorized into glioma, meningioma, pituitary, and no tumor. The images are split into training and test sets.

Download the MRI brain tumor dataset.  You can find the dataset [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data).

Source 1: [Figshare](https://figshare.com/articles/dataset/brain_tumour_dataset/1512427) - 3064 T1-weighted contrast-enhanced images from 233 patients.

Source 2: [Kaggle](https://www.kaggle.com/sartajbhuvaji/brain-tumour-classification-mri/metadata) - Brain Tumor Classification MRI - 3264 images split into training and test data.

Source 3: [Kaggle](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumour-detection/metadata) - Brain Tumor Detection - 3865 images labeled as yes or no based on the presence of a tumor.

## Technologies Used
1. Python
2. TensorFlow
3. Keras
4. PyTorch and Torch Geometric
5. Spektral
6. Scikit-learn
7. NumPy
8. Pandas
9. Matplotlib
10. Seaborn 

## Prerequisites
Python 3.8 or higher
GPU (recommended for training deep learning models)

## Project Structure
1. Data Preparation:

The dataset is extracted from a zip file and organized into training and testing directories.
Image data is preprocessed using various transformations.

2. Model Implementation:

Custom models (GCN and GAN) are implemented alongside pre-trained models (VGG19 and ResNet50).
Data augmentation and regularization techniques like L2 regularization, dropout, and callbacks (early stopping, learning rate reduction) are employed.

3. Training and Evaluation:

The models are trained on the dataset, and performance is evaluated using accuracy, precision, recall, F1-score, and ROC-AUC metrics.
The results are compared to determine the most effective model.

## Results
The project aims to identify the most effective model for brain tumor classification based on the selected evaluation metrics. The detailed results and comparative analysis will be included in the final report.

## Contributors
Sahana Muralidaran (sm22adg@herts.ac.uk)
## License
The dataset used in this project is licensed under the MIT License - see the [LICENSE](https://www.mit.edu/~amini/LICENSE.md) file for details.
