# -*- coding: utf-8 -*-
"""ProjetoBimestral2-PDI.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YN4_CPjJ7tNMbF_8h8EGo_9IP92SGTna

Import libraries
"""

from google.colab import drive
drive.mount('/content/drive')

import os

path = '/content/drive/MyDrive/UTFPR/Cars Detection'

os.chdir(path)
!ls

import os
import glob as glob
import matplotlib.pyplot as plt
import cv2
import requests
import random
import numpy as np

def yolo2standard(bboxes):
     xmin = bboxes[0]-bboxes[2]/2
     xmax = bboxes[0]+bboxes[2]/2
     ymin = bboxes[1]-bboxes[3]/2
     ymax = bboxes[1]+bboxes[3]/2
     return xmin, ymin, xmax, ymax

class_names = [ 'Ambulance', 'Bus', 'Car','Motorcycle', 'Truck']
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

def plot_box(image, bboxes, labels):
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2standard(box)
        xmin = int(x1*w)
        xmax = int(x2*w)
        ymin = int(y1*h)
        ymax = int(y2*h)
        width = xmin - xmax
        height = ymin - ymax

        class_name = class_names[int(labels[box_num])]

        cv2.rectangle(
            image,
            (xmin, ymin), (xmax, ymax),
            color=colors[class_names.index(class_name)],
            thickness=2
        )

        font_scale = min(1,max(3,int(w/500)))
        font_thickness = min(2, max(10,int(w/50)))

        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))

        # Text width and height
        tw, th = cv2.getTextSize(class_name, 0, fontScale=font_scale, thickness=font_thickness)[0]
        p2 = p1[0] + tw, p1[1] + -th - 10
        cv2.rectangle(
            image,
            p1, p2,
            color=colors[class_names.index(class_name)],
            thickness=-1,
        )
        cv2.putText(
            image,
            class_name,
            (xmin+1, ymin-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    return image

def plot(image_paths, label_paths, num_samples):
    all_training_images = glob.glob(image_paths)
    all_training_labels = glob.glob(label_paths)
    all_training_images.sort()
    all_training_labels.sort()

    num_images = len(all_training_images)

    if num_images == 0 or len(all_training_labels) == 0:
        print("Não há imagens ou rótulos disponíveis para plotagem.")
        return

    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = random.randint(0, num_images - 1)
        if j >= len(all_training_labels):
            print(f"Índice {j} fora do alcance da lista de rótulos.")
            continue

        image = cv2.imread(all_training_images[j])
        with open(all_training_labels[j], 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.show()

plot(image_paths='/content/drive/MyDrive/UTFPR/Cars Detection/train/images/*',
    label_paths='/content/drive/MyDrive/UTFPR/Cars Detection/train/labels/*',
    num_samples=4,
)

# Commented out IPython magic to ensure Python compatibility.
# Clone o repositório do YOLOv8
!git clone https://github.com/ultralytics/yolov5.git
# %cd yolov5

# Instale as dependências
!pip install -U -r requirements.txt

# Crie o arquivo de configuração para o seu conjunto de dados (data.yaml)
data_yaml_content = """
path: /content/drive/MyDrive/UTFPR/Cars Detection
train: train/images
val: valid/images

nc: 5
names: ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
"""

with open('data.yaml', 'w') as f:
    f.write(data_yaml_content)

# Treine o modelo
!python train.py --img-size 640 --batch-size 16 --epochs 30 --data data.yaml --cfg yolov5s.yaml --name cars_detection

# Para fazer a detecção em uma imagem de exemplo
!python detect.py --source path/to/your/test/image.jpg --weights runs/train/cars_detection/weights/best.pt --conf 0.5