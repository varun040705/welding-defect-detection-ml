import os
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import time

base_path = r"C:\Users\KML\Downloads\archive\The Welding Defect Dataset - v2\The Welding Defect Dataset - v2"
image_path = os.path.join(base_path, "train", "images")
label_path = os.path.join(base_path, "train", "labels")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

st.title("Welding Defect Classifier")

slideshow_placeholder = st.empty()
progress_bar = st.progress(0)
status_text = st.empty()

all_images = sorted([f for f in os.listdir(image_path) if f.endswith('.jpg')])
slideshow_images = all_images[:10]

features, labels = [], []

for i, img_file in enumerate(all_images):
    status_text.text(f"Processing image {i + 1} of {len(all_images)}: {img_file}")

    current_slideshow_img = slideshow_images[i % len(slideshow_images)]
    img_for_slide = Image.open(os.path.join(image_path, current_slideshow_img)).convert('RGB')
    slideshow_placeholder.image(img_for_slide, width=300)

    img_path = os.path.join(image_path, img_file)
    label_file = os.path.join(label_path, img_file.replace('.jpg', '.txt'))
    if not os.path.exists(label_file):
        continue
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = resnet(img_tensor).squeeze().cpu().numpy()
    with open(label_file, 'r') as f:
        line = f.readline()
        if not line.strip():
            continue
        class_id = int(line.strip().split()[0])
        features.append(feature)
        labels.append(class_id)

    progress_bar.progress((i + 1) / len(all_images))

slideshow_placeholder.empty()
status_text.text("Feature extraction completed")
progress_bar.empty()

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
st.text("Classification Report:\n" + classification_report(y_test, y_pred))

class_names = {0: "CORRECT", 1: "DEFECTIVE"}

uploaded_file = st.file_uploader("Upload a welding image to test", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        test_feature = resnet(img_tensor).squeeze().cpu().numpy().reshape(1, -1)
    prediction = model.predict(test_feature)[0]
    confidence = np.max(model.predict_proba(test_feature))
    
    st.write(f"Prediction: The welding process is {class_names[prediction]}")
    st.write(f"Confidence: {confidence:.2f}")
