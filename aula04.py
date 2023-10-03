import os
import cv2
import csv
import numpy as np
from scipy.stats import describe

def calculate_glcm(image_gray):
    # Calculo GLCM (propriedades: contraste, dissimilaridade, homogeneidade, ASM)
    # Use o calcHist do OpenCV para calcular o GLCM
    glcm = cv2.calcHist([image_gray.astype(np.uint8)], [0], None, [256], [0, 256])
    glcm /= glcm.sum()  # Normalize para obter as probabilidades
    contrast = np.sum(glcm * (np.arange(256) ** 2)) - (np.sum(glcm * np.arange(256))) ** 2
    dissimilarity = np.sum(glcm * np.abs(np.arange(256)[:, None] - np.arange(256)))
    homogeneity = np.sum(glcm / (1. + np.abs(np.arange(256) - np.arange(256)))).item()
    asm = np.sum(glcm ** 2).item()
    properties = {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'ASM': asm
    }
    return properties

def extract_image_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Extraindo características estruturais (largura e altura)
    height, width = image.shape[:2]

    # Calculando estatisticas
    stats = describe(image.ravel())

    # Calculandao caracteristicas GLCM
    glcm_properties = calculate_glcm(image)

    return {'width': width, 'height': height}, stats, glcm_properties

def list_image_features_in_directory(directory):
    try:
        if os.path.isdir(directory):
            files = [f for f in os.listdir(directory) if f.lower().endswith(('.pgm', 'png', '.jpg', '.jpeg', '.gif'))]
            print("Image Features:")
            for file in files:
                image_path = os.path.join(directory, file)
                features = extract_image_features(image_path)
                print(f"Image: {file}")
                print("Structural Characteristics:")
                print(features[0])
                print("Statistics:")
                print(features[1])
                print("GLCM Properties:")
                print(features[2])
                print("--------------------")
        else:
            print("Diretorio Invalido")
    except Exception as e:
        print(f"Erro: {str(e)}")

directory_path = 'imagens/imgMaiusculas/MAIUSCULAS'
output_directory_path = ''
list_image_features_in_directory(directory_path)