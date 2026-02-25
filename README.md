Welding Defect Detection using Machine Learning
Overview

This project presents an intelligent system for automatic detection and classification of welding defects using Machine Learning and Computer Vision. The application allows users to upload welding images and get real-time predictions of defects through a user-friendly web interface.

The system helps reduce manual inspection, improves accuracy, and ensures safety and quality in industrial manufacturing processes.

 Features

✔ Upload welding images

✔ Detect and classify multiple welding defects

✔ Real-time prediction

✔ Simple and interactive UI

✔ High accuracy model

✔ Scalable for industrial applications

✔ Can be extended for real-time monitoring

 Technologies Used

Python

Machine Learning

Computer Vision

OpenCV

NumPy

Pandas

Scikit-learn / Deep Learning

Streamlit (Web interface)

 Dataset

The dataset contains labeled welding images including:

Crack

Porosity

Lack of fusion

Slag inclusion

Good weld samples

Images were preprocessed and augmented to improve model performance.

 System Workflow

1️⃣ Image Upload

2️⃣ Image Preprocessing

3️⃣ Feature Extraction

4️⃣ Model Prediction

5️⃣ Defect Classification

6️⃣ Result Display on Streamlit UI

 Installation
Step 1: Clone the repository
git clone https://github.com/your-username/welding-defect-detection.git
cd welding-defect-detection
Step 2: Install required libraries
pip install -r requirements.txt
Step 3: Run the application
streamlit run app.py
 Model Details

Images resized and normalized

Data augmentation applied

Supervised learning approach

Model evaluated on unseen test images

Performance monitored using accuracy and loss metrics

 Results

The trained model achieved strong performance in detecting welding defects and demonstrated good generalization on test data. The system effectively identifies different defect types and provides quick predictions.
