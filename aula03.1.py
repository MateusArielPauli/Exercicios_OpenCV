import cv2
import numpy as np

# Carregue a imagem
image = cv2.imread('./Imagens/galinha.png')

# Define as coordenadas do quadrado vermelho (x, y, largura, altura)
x = 178  # Coordenada x do canto superior esquerdo do quadrado vermelho
y = 100  # Coordenada y do canto superior esquerdo do quadrado vermelho
width = 70  # Largura do quadrado vermelho
height = 66  # Altura do quadrado vermelho


# Verifique se as coordenadas estão dentro dos limites da imagem
if x >= 0 and y >= 0 and x + width <= image.shape[1] and y + height <= image.shape[0]:
    # Crie uma região de interesse (ROI) dentro do quadrado vermelho
    roi = image[y:y + height, x:x + width]

    # Converta a ROI para tons de cinza
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Aplique um limiar adaptativo para binarizar a imagem (fazer com que as galinhas se destaquem)
    binary = cv2.adaptiveThreshold(roi_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,41,2)

    # definindo um kernel de tamanho 3x3 para aplicar as transformações
    kernel = np.ones((3,3), np.uint8) 
    
    # realizando a abertura da imagem vizando remover alguns ruidos
    binary = cv2.erode(binary, kernel, iterations=4) 
    binary = cv2.dilate(binary, kernel, iterations=4)
    
    # Encontrando contornos na imagem binarizada
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Desenha os contornos na ROI original
    cv2.drawContours(roi, contours, -1, (0, 255, 0), 1)  # Contornos verdes

    # Conta o número de galinhas (contornos) encontrados
    num_galinhas = len(contours)

    # Desenha o resultado na imagem original
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)  # Quadrado vermelho
    cv2.putText(image, f'Galinhas: {num_galinhas}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Número de galinhas

    # Exiba a imagem com o resultado
    cv2.imshow('Resultado', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Exiba o número de galinhas encontradas
    print(f'Número de galinhas encontradas: {num_galinhas}')
else:
    print("As coordenadas estão fora dos limites da imagem.")