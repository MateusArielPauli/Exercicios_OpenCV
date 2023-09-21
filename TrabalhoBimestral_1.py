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
import cv2
import numpy as np

class PdiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_image = None
        self.setWindowTitle("Trabalho Bimestral")
        self.setGeometry(100, 100, 300, 500)

        # Layout principal
        main_layout = QVBoxLayout()

        open_button = QPushButton("Abrir Imagem", self)
        open_button.clicked.connect(self.open_image)
        main_layout.addWidget(open_button)


        # Botões para ações
        convert_button = QPushButton("Converção de cor: HLS", self)
        convert_button.clicked.connect(self.apply_conversion)
        main_layout.addWidget(convert_button)

        median_button = QPushButton("Filtro: Mediana", self)
        median_button.clicked.connect(self.apply_median_filter)
        main_layout.addWidget(median_button)

        laplace_button = QPushButton("Detector de Bordas: Laplace", self)
        laplace_button.clicked.connect(self.apply_laplace_edge_detection)
        main_layout.addWidget(laplace_button)

    
        # Label para exibir a imagem
        self.image_label = QLabel(self)
        main_layout.addWidget(self.image_label)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def apply_conversion(self):
        if self.selected_image is not None:
            hls_image = cv2.cvtColor(self.selected_image, cv2.COLOR_BGR2HLS)
            self.display_image(hls_image)

    def apply_median_filter(self):
        if self.selected_image is not None:
            filtered_image = cv2.medianBlur(self.selected_image, 15)
            self.display_image(filtered_image)

    def apply_laplace_edge_detection(self):
        if self.selected_image is not None:
            edges = cv2.Laplacian(self.selected_image, cv2.CV_64F)
            edges = cv2.convertScaleAbs(edges)
            self.display_image(edges)

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