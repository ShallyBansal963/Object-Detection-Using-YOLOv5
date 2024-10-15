# Object Detection Using YOLO Object Detector

This project demonstrates object detection in images Deep Learning, OpenCV, and Python. I utilize the **YOLOv5** (You Only Look Once version 5) model, specifically trained on the **COCO dataset**.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Analysis](#data-analysis)
- [Preprocessing](#preprocessing)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Screenshots](#screenshots)
- [Cloning YOLOv5 Repository](#cloning-yolov5-repository)

## Overview

The **COCO (Common Objects in Context)** dataset contains 80 labels, including but not limited to:

- Bicycles
- Cars and trucks
- Airplanes
- Stop signs and fire hydrants
- Various animals (e.g., cats, dogs, birds, horses, cows, sheep)
- Kitchen and dining objects (e.g., wine glasses, forks, knives, spoons)

For a complete list of what YOLO can detect, please refer to the [COCO dataset website](http://cocodataset.org/#home).

## Dataset

We download the COCO dataset, which consists of images and annotations, directly from the official COCO website. The following Python script facilitates the download and extraction of the dataset:

# Object Detection Using YOLO Object Detector

This project demonstrates object detection in both images and video streams using Deep Learning, OpenCV, and Python. We utilize the **YOLOv3** (You Only Look Once version 3) model, specifically trained on the **COCO dataset**.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Analysis](#data-analysis)
- [Preprocessing](#preprocessing)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Screenshots](#screenshots)
- [Cloning YOLOv5 Repository](#cloning-yolov5-repository)

## Overview

The **COCO (Common Objects in Context)** dataset contains 80 labels, including but not limited to:

- Bicycles
- Cars and trucks
- Airplanes
- Stop signs and fire hydrants
- Various animals (e.g., cats, dogs, birds, horses, cows, sheep)
- Kitchen and dining objects (e.g., wine glasses, forks, knives, spoons)

For a complete list of what YOLO can detect, please refer to the [COCO dataset website](http://cocodataset.org/#home).

## Dataset

We download the COCO dataset, which consists of images and annotations, directly from the official COCO website. The following Python script facilitates the download and extraction of the dataset:

import os
import requests
import zipfile
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# COCO dataset URLs
coco_images_url = "http://images.cocodataset.org/zips/train2017.zip"
coco_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
save_dir = "./data/COCO"

def download_and_extract(url, save_dir, file_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file_name)
    if not os.path.exists(file_path):
        response = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(1024)):
                f.write(chunk)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)

def split_dataset(dataset_dir, test_size=0.2):
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)
    return train_files, test_files

# Downloading COCO dataset
download_and_extract(coco_images_url, save_dir, "train2017.zip")
download_and_extract(coco_annotations_url, save_dir, "annotations_trainval2017.zip")

# Split dataset into training and testing sets
train_files, test_files = split_dataset(os.path.join(save_dir, 'train2017'))
print(f"Training files: {len(train_files)}, Testing files: {len(test_files)}")

Data Analysis

We can analyze the dataset annotations and visualize the distribution of object categories with the following code snippet:

python
Copy code
import json
import matplotlib.pyplot as plt
from collections import Counter

# Load COCO annotations
with open('./data/COCO/annotations/instances_train2017.json') as f:
    annotations = json.load(f)

# Create a mapping of category IDs to category names
category_mapping = {cat['id']: cat['name'] for cat in annotations['categories']}
category_ids = [ann['category_id'] for ann in annotations['annotations']]
category_counts = Counter(category_ids)

# Generate a bar chart for the top object categories
category_names = [category_mapping[cat_id] for cat_id in category_counts.keys()]
category_values = list(category_counts.values())

plt.figure(figsize=(12, 8))
plt.barh(category_names, category_values, color='skyblue')
plt.title('Distribution of Object Categories in COCO Dataset')
plt.xlabel('Number of Instances')
plt.ylabel('Category')
plt.show()
Preprocessing

The images will undergo preprocessing to fit the YOLO model's input requirements. The following pipeline resizes and augments the images:

from PIL import Image
from torchvision import transforms

# Preprocessing pipeline for COCO images
preprocess = transforms.Compose([
    transforms.Resize((416, 416)),  # Resize image to 416x416 (YOLO input size)
    transforms.RandomHorizontalFlip(),  # Data augmentation: random flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color adjustments
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (for ImageNet pre-trained models)
])
Installation

Before running the project, ensure the required libraries installed. We can install them using pip:

pip install numpy
pip install opencv-python

To perform object detection, use the following command in your terminal:

python yolo.py --image images/baggage_claim.jpg
Screenshots

Here are some results showcasing YOLO's object detection capabilities:


To clone the YOLOv5 repository, execute the following command:


git clone https://github.com/ultralytics/yolov5  # Clone YOLOv5 repository
