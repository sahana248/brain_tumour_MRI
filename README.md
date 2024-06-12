# Comparative Analysis of deep learning models for MRI- Based brain tumour Classification.
## Research Question
How does the performance of custom deep learning models (GCN and GAN) compare to pretrained models (VGG19 and ResNet50) for brain tumour classification?

## Project Objectives
In this project, we aim to develop and evaluate deep learning models, including Graph Convolutional Networks (GCN) and Generative Adversarial Networks (GAN), for brain tumour classification using MRI images. Our approach involves training and fine-tuning these custom models on a labelled dataset of brain MRI scans. We will then compare their performance with pretrained models such as VGG19 and ResNet50. Evaluation metrics, including accuracy, precision, recall, F1-score, and ROC-AUC, will guide us in determining the effectiveness of each model.

## Summary and Background
This research investigates the efficacy of deep learning techniques applied to magnetic resonance imaging for accurate diagnosis of brain tumours. The detection of brain tumours is crucially important but it is generally difficult as their recognition requires trained experts. Consequently, this necessitates machine-aided approaches since manual examination is time-consuming and fraught with mistakes. 

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

## Prerequisites
Python 3.8 or higher
GPU (recommended for training deep learning models)
## Dataset
Download the MRI brain tumor dataset.  You can find the dataset [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data).
## Contributors
Sahana Muralidaran (sm22adg@herts.ac.uk)
## License
The dataset used in this project is licensed under the MIT License - see the [LICENSE](https://www.mit.edu/~amini/LICENSE.md) file for details.
