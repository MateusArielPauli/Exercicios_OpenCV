import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QSlider
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class PdiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_image = None
        self.setWindowTitle("Trabalho Bimestral")
        self.setGeometry(100, 100, 300, 300)

        # Layout principal
        main_layout = QVBoxLayout()

        # Botões para ações
        convert_button = QPushButton("Converção de cor: HLS", self)
        convert_button.clicked.connect(self.apply_conversion)
        main_layout.addWidget(convert_button)
        convert_button.setStyleSheet("font-size: 16px")  # Ajuste o tamanho da fonte conforme necessário

        main_layout.addSpacing(50)
        
        median_button = QPushButton("Filtro: Mediana", self)
        median_button.clicked.connect(self.apply_median_filter)
        main_layout.addWidget(median_button)
        median_button.setStyleSheet("font-size: 16px")  # Ajuste o tamanho da fonte conforme necessário

        # Slider para controlar a intensidade do filtro blur
        blur_slider = QSlider(Qt.Horizontal, self)
        blur_slider.setMinimum(1)  # Valor mínimo do slider
        blur_slider.setMaximum(30)  # Valor máximo do slider
        blur_slider.setValue(15)  # Valor inicial do slider
        blur_slider.valueChanged.connect(self.update_blur_intensity)
        main_layout.addWidget(blur_slider)

        main_layout.addSpacing(50)

        laplace_button = QPushButton("Detector de Bordas: Laplace", self)
        laplace_button.clicked.connect(self.apply_laplace_edge_detection)
        main_layout.addWidget(laplace_button)
        laplace_button.setStyleSheet("font-size: 16px")  # Ajuste o tamanho da fonte conforme necessário

        main_layout.addSpacing(50)

        binary_button = QPushButton("Binarizar imagem: Threshould", self)
        binary_button.clicked.connect(self.apply_binarization)
        main_layout.addWidget(binary_button)
        binary_button.setStyleSheet("font-size: 16px")  # Ajuste o tamanho da fonte conforme necessário

        main_layout.addSpacing(50)

        erosion_button = QPushButton("Morfologia Matemática: Erosão", self)
        erosion_button.clicked.connect(self.apply_erosion)
        main_layout.addWidget(erosion_button)
        erosion_button.setStyleSheet("font-size: 16px")  # Ajuste o tamanho da fonte conforme necessário

        main_layout.addSpacing(50)

        # Adicionar espaçamento entre os botões
        main_layout.addSpacing(80)
        open_button = QPushButton("Abrir Imagem", self)
        open_button.clicked.connect(self.open_image)
        main_layout.addWidget(open_button)
        open_button.setStyleSheet("font-size: 16px")  # Ajuste o tamanho da fonte conforme necessário


        reset_button = QPushButton("Reverter para Original", self)
        reset_button.clicked.connect(self.reset_image)
        main_layout.addWidget(reset_button)
        reset_button.setStyleSheet("font-size: 16px")  # Ajuste o tamanho da fonte conforme necessário
    
        # Label para exibir a imagem
        self.image_label = QLabel(self)
        main_layout.addWidget(self.image_label)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def update_blur_intensity(self, value):
        # Garantir que o valor seja ímpar
        self.blur_intensity = value if value % 2 == 1 else value + 1

    def apply_conversion(self):
        if self.selected_image is not None:
            hls_image = cv2.cvtColor(self.selected_image, cv2.COLOR_BGR2HLS)
            self.display_image(hls_image)

    def apply_median_filter(self):
        if self.selected_image is not None:
            filtered_image = cv2.medianBlur(self.selected_image, self.blur_intensity)
            self.display_image(filtered_image)

    def apply_laplace_edge_detection(self):
        if self.selected_image is not None:
            edges = cv2.Laplacian(self.selected_image, cv2.CV_64F)
            edges = cv2.convertScaleAbs(edges)
            self.display_image(edges)

    def apply_binarization(self):
        if self.selected_image is not None:
            # Aplicar binarização com um limiar (threshold)
            gray_image = cv2.cvtColor(self.selected_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            self.display_image(binary_image)

    def apply_erosion(self):
        if self.selected_image is not None:
            # Aplicar operação de erosão
            gray_image = cv2.cvtColor(self.selected_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            erosion_image = cv2.erode(binary_image, kernel, iterations=1)
            self.display_image(erosion_image)

    def reset_image(self):
        if self.selected_image is not None:
            self.display_image(self.selected_image)

    def display_image(self, image_copy):
        if image_copy is not None:
            cv2.imshow("Image", image_copy) 

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp *.gif);;All Files(*)", options=options)

        if image_path:
            try:
                self.selected_image = cv2.imread(image_path)
                self.display_image(self.selected_image)
            except Exception as e:
                print("Erro ao abrir a imagem:", str(e))
        else:
            print("Nenhuma imagem selecionada.")
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = PdiApp()
    main_window.show()
    sys.exit(app.exec_())