import cv2
import numpy as np

# Passo 1: carrega a imagem
img = cv2.imread('./Imagens/bocha.jpg')

# Passo 2: Converta a imagem para HSV para trabalhar com as cores
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Passo 3: Aplique um filtro de suavização (desfoque gaussiano)
blur = cv2.GaussianBlur(hsv, (3, 3), 0)

# Passo 4: Use a detecção de bordas (Canny)
# Os parametros de H foram definidos a partir da cor de fundo
bordas = cv2.Canny(blur, 95, 130)

# Passo 5: Aplique uma transformação de dilatação
kernel = np.ones((3, 3), np.uint8)
imagem_dilatada = cv2.dilate(bordas, kernel, iterations=2)

# Passo 6: Encontre os contornos
contornos, _ = cv2.findContours(imagem_dilatada.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Passo 7: Desenhe os contornos na imagem binária em branco (255,255,255)
imagem_binaria = np.zeros_like(hsv)
cv2.drawContours(imagem_binaria, contornos, -1,(255,255,255), thickness=cv2.FILLED)

# Exibir a imagem binária 
cv2.imshow('Bochas detectadas', np.hstack([img, imagem_binaria]))
cv2.waitKey(0)