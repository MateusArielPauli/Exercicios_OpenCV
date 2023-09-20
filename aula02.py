import cv2
import numpy as np

# Variáveis globais para os valores dos TrackBars
low_pass_filter = 0
high_pass_filter = 0

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Captura a cor do pixel no local do clique
        color = frame[y, x]
        print(f"Cor na posição ({x}, {y}): {color}")

def update_low_pass_filter(value):
    global low_pass_filter
    low_pass_filter = value
    apply_filters()

def update_high_pass_filter(value):
    global high_pass_filter
    high_pass_filter = value
    apply_filters()

def apply_filters():
    # Aplica os filtros passa-baixa
    kernel_size = 2 * low_pass_filter + 1
    blurred = cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)

    # Aplica o filtro passa-alta
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)

    # Binariza a imagem
    _, binary = cv2.threshold(sharpened, high_pass_filter, 255, cv2.THRESH_BINARY)

    # Exibe a imagem
    cv2.imshow('Result', binary)

# Inicializa a câmera
cap = cv2.VideoCapture(0)

# Verifica se a câmera foi aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Cria uma janela para exibir a imagem
cv2.namedWindow('Camera Image')
cv2.setMouseCallback('Camera Image', on_mouse_click)

# Cria as trackbars
cv2.createTrackbar('Low Pass Filter', 'Camera Image', low_pass_filter, 10, update_low_pass_filter)
cv2.createTrackbar('High Pass Filter', 'Camera Image', high_pass_filter, 255, update_high_pass_filter)

while True:
    # Captura um frame da câmera
    ret, frame = cap.read()

    # Converte o frame para tons de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplica os filtros
    apply_filters()

    # Exibe o frame original
    cv2.imshow('Camera Image', frame)

    # Sai do loop quando a tecla 'q' é pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()