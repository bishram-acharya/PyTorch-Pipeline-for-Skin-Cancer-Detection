# Skin Cancer Classification using Deep Learning (ISIC 2019 Challenge)

## 1. Introduction
Skin cancer is a critical global health concern, with millions of cases diagnosed annually. Early and accurate detection is essential for effective treatment and better patient outcomes. Dermoscopic imaging is a powerful, non-invasive tool that provides detailed visualizations of skin layers. However, diagnosing skin cancer from dermoscopic images can be challenging, requiring expertise and often prone to human error. This project aims to build a machine learning model to assist clinicians in the analysis and diagnosis of skin lesions.

## 2. ISIC 2019 Challenge
The ISIC 2019 Challenge provides a large dataset of dermoscopic images, enabling the development of machine learning models to classify skin lesions. The challenge focuses on categorizing images into nine diagnostic categories, including conditions like melanoma and basal cell carcinoma, as well as benign lesions. Accurate classification is vital to help dermatologists make more informed decisions regarding further investigation and potential biopsies.

## 3. Dataset Description
The ISIC 2019 dataset consists of 25,331 dermoscopic images classified into the following nine categories:
- Seborrheic Keratosis
- Vascular Lesion
- Basal Cell Carcinoma
- Melanoma
- Squamous Cell Carcinoma
- Pigmented Benign Keratosis
- Dermatofibroma
- Nevus
- Actinic Keratosis

## 4. Objective
The objective of this project is to build a deep learning model using PyTorch to classify dermoscopic images into one of the nine categories from the ISIC 2019 dataset. This project aims to develop expertise in building, training, and evaluating deep learning models, with insights transferable to similar projects.

## 5. Performance Metrics
To evaluate the model's effectiveness, we will use the following metrics:
- Balanced Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## 6. Challenges
- **Class Imbalance**: The dataset contains more benign lesions than malignant ones, making it difficult for the model to accurately classify rare yet dangerous conditions like melanoma. We will address this using weighted loss functions or data augmentation.
- **Model Overfitting**: Due to the large dataset, the model may overfit. We will use techniques such as dropout, regularization, and early stopping to mitigate overfitting.
- **Limited GPU Resources**: To optimize GPU use, we will implement efficient training techniques like learning rate scheduling and batch size adjustments.

## 7. Advantages of Applying ML to the Problem
### 7.1 Time Efficiency
Machine learning models can classify skin lesions quickly once trained, greatly reducing the time required for diagnosis. This allows dermatologists to focus on complex cases and prioritize patients with potentially severe conditions, enhancing clinical efficiency.

### 7.2 Cost Reduction
By accurately identifying benign lesions, ML models can reduce unnecessary biopsies, saving costs for both patients and healthcare providers. Automating the initial screening process also lowers the workload on healthcare professionals, reducing staffing needs and operational costs.

## 8. Resources
### 8.1 GPU and Dataset
- Kaggle’s free access to NVIDIA TESLA P100 GPUs for training and testing.
- Publicly available ISIC 2019 dataset.

### 8.2 Libraries
- PyTorch
- Tqdm
- Pillow
- OS module
- Sklearn

## 9. Plan of Action
1. Data Loading and Exploration
2. Data Preprocessing
3. Model Architecture
4. Training
5. Evaluation
6. Results Analysis
7. Conclusion
8. Recommendations for Future Improvements
