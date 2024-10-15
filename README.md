# Object Detection Using YOLOv5

This project implements object detection in both images and video streams using Deep Learning, OpenCV, and Python, specifically employing the YOLOv5 model. YOLOv5 is known for its speed and accuracy in detecting objects in real-time, making it an excellent choice for various applications.

## Overview

In this project, we utilize the YOLOv5 model to detect and classify objects in images. The model is pre-trained on the COCO (Common Objects in Context) dataset, which contains a diverse range of images and annotations. The dataset comprises 80 object categories, including but not limited to:

- People
- Bicycles
- Cars and trucks
- Airplanes
- Stop signs and fire hydrants
- Animals (e.g., cats, dogs, birds, horses, cows, sheep)
- Kitchen and dining items (e.g., wine glasses, cups, forks, knives, spoons)

You can find the full list of categories that YOLO is trained to detect in the [COCO dataset documentation](https://cocodataset.org/#home).

## Dataset

The COCO dataset is used for training and testing the model. It is a large-scale dataset designed for various visual recognition tasks. The dataset provides images with annotations for multiple objects, which allows for effective training of the YOLOv5 model.

## Installation

To run this project, you'll need to install the necessary dependencies. You can do this by running the following commands:

pip install numpy
pip install opencv-python
pip install torch torchvision torchaudio
pip install matplotlib

## Running the Project

After setting up the environment, you can run the object detection model on an image by executing:

python yolo.py --image path/to/your/image.jpg
You can also modify the yolo.py script to process video streams or batches of images, depending on your application needs.

## YOLOv5 Model Files

This project uses the pre-trained YOLOv5 model files provided by the Ultralytics team. You can find the YOLOv5 repository here to explore additional features and capabilities.

## Conclusion

This object detection project demonstrates the capabilities of the YOLOv5 model in identifying and localizing various objects in images and video streams. The results highlight YOLOv5's efficiency and accuracy, making it a powerful tool for real-time object detection applications.

## Acknowledgments

Special thanks to the authors of the YOLOv5 model and the creators of the COCO dataset for their invaluable contributions to the field of computer vision.



