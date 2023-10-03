import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import csv
import os

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Características estruturais
    # Exemplo: Área e Perímetro (podem não ser precisos, dependendo da imagem)
    ret,thresh = cv2.threshold(image,127,255,0)
    contours,_ = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    area = cv2.contourArea(cnt) # Área
    perimeter = cv2.arcLength(cnt,True) # Perímetro
    
    # Estatísticas
    mean = np.mean(image) # Média dos pixels
    std_dev = np.std(image) # Desvio padrão dos pixels
    max_value = np.max(image) # Valor máximo de pixel
    min_value = np.min(image) # Valor mínimo de pixel
    
    # Características GLCM
    glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    return area, perimeter, mean, std_dev, max_value, min_value, contrast, dissimilarity, homogeneity, energy

def process_directory(directory):
    with open('extracaoCaracteristicas.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escrevendo o cabeçalho do CSV
        writer.writerow(['Class_Name', 'Area', 'Perimeter', 'Mean', 'Std_Dev', 'Max_Value', 'Min_Value', 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy'])

        for filename in os.listdir(directory):
            if filename.endswith(".pgm") or filename.endswith(".jpg"):
                filepath = os.path.join(directory, filename)
                area, perimeter, mean, std_dev, max_value, min_value, contrast, dissimilarity, homogeneity, energy = extract_features(filepath)
                
                # Supondo que o nome da classe seja derivado do nome do arquivo
                class_name = filename.split('_')[0]
                print(f"{class_name}, {area}, {perimeter}, {mean}, {std_dev}, {max_value}, {min_value}, {contrast}, {dissimilarity}, {homogeneity}, {energy}")
                # Escrevendo as características extraídas no arquivo CSV
                writer.writerow([class_name, area, perimeter, mean, std_dev, max_value, min_value, contrast, dissimilarity, homogeneity, energy])
process_directory('imagens/imgMaiusculas/MAIUSCULAS/')