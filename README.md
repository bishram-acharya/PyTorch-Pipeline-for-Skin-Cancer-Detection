1 Introduction
Skin cancer is a major global health issue, with millions of new cases diagnosed annually. Early
and accurate detection is critical for successful treatment and improved patient outcomes. One
of the most effective methods for diagnosing skin cancer is through dermoscopic imaging, a noninvasive technique that allows for the visualization of deeper layers of the skin. However, diagnosing
skin cancer through non-invasive imaging techniques requires thorough image analysis by expert
clinicians, and prone to human error. Consequently, significant efforts have been made in recent
years to develop tools that assist physicians in analyzing dermoscopic images.
2 ISIC 2019 Challenge
The ISIC 2019 Challenge aims to address this problem by providing a large dataset of dermoscopic
images to train machine learning models for the classification of skin lesions. The goal is to categorize
images into one of nine diagnostic categories, including serious conditions such as melanoma and
basal cell carcinoma, as well as more benign lesions. Accurate classification is crucial because it
can assist dermatologists in making more informed decisions about whether a lesion requires further
investigation, such as a biopsy.
3 Dataset Description
The dataset for ISIC 2019 contains 25,331 images available for the classification of dermoscopic
images among nine different diagnostic categories:
• Seborrheic Keratosis
• Vascular Lesion
• Basal Cell Carcinoma
• Melanoma
• Squamous Cell Carcinoma
• Pigmented Benign Keratosis
• Dermatofibroma
• Nevus
• Actinic Keratosis
4 Objective
The objective of this project is to build a deep learning model using PyTorch that can classify
dermoscopic images into one of the nine categories mentioned in the ISIC 2019 dataset. The focus
is not only on solving this problem in particular but also on understanding the process of building,
training, and evaluating the model so that the acquired knowledge can be transferable to future
projects.
3
5 Performance Metrics
The following metrics will be used to evaluate the model: - Balanced Accuracy, Precision,
Recall, F1-Score, ROC-AUC
6 Challenges
• Class Imbalance: The dataset contains many more benign lesions than malignant ones. This
imbalance can make it difficult for the model to learn to correctly classify rare but dangerous
categories like melanoma. This will be addressed using techniques such as weighted loss
functions or data augmentation.
• Model Overfitting: With a large number of images and possible overfitting, we will explore
techniques such as dropout, regularization, and early stopping.
• Limited GPU Time: Since the GPU resource is limited, the model will be optimized to
converge efficiently using techniques like learning rate scheduling and batch size adjustments.
7 Advantages of applying ML to the problem
7.0.1 Time Efficiency:
Manual examination of dermoscopic images by dermatologists is a time-consuming process, especially when handling large volumes of cases. Machine learning models, once trained, can analyze and
classify skin lesions almost instantaneously, greatly reducing the time spent per diagnosis. This enables dermatologists to focus on more complex cases and prioritize patients with potentially serious
conditions, improving overall efficiency in clinical settings.
7.0.2 Cost Reduction:
The use of ML for skin lesion classification can lead to substantial cost savings by reducing the
need for unnecessary biopsies. ML models can help in accurately identifying benign lesions, sparing
patients from costly and invasive procedures. Furthermore, automating initial screenings of dermoscopic images can lower the workload on healthcare professionals, allowing hospitals and clinics
to operate more cost-effectively by minimizing the need for additional staffing to handle large case
volumes.
8 Resources
8.1 GPU and dataset :
• Kaggle’s free access to NVIDIA TESLA P100 GPUs for training and testing.
• Publicly available ISIC 2019 dataset ## Libraries :
• PyTorch
• Tqdm
• Pillow
• os module
• sklearn
4
9 Plan of Action
1. Data Loading and Exploration
2. Data Preprocessing
3. Model Architecture
4. Training
5. Evaluation
6. Results Analysis
7. Conclusion
8. Recommendations for Future Improvements
