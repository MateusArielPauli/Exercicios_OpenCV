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

class Action:
    def __init__(self, nome, *args,):
        self.nome = nome
        self.args = args

class PdiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_image = None
        self.selected_elements = []
        self.selected_action = None

        self.default_actions =  [
            Action("Converter para Cinza", 0),
            Action("Filtro: Mediana", 0), 
            Action("Detectar Bordas - Canny", 0, 0), 
            Action("Binarizar Imagem"), 
            Action("Morfologia Matemática - erosão")
        ]
        
        self.setWindowTitle("Trabalho Bimestral")
        self.setGeometry(100, 100, 300, 500)

        # Layout principal
        main_layout = QHBoxLayout()

        # Área lateral para escolher elementos
        self.lateral_widget = QWidget()
        self.lateral_layout = QVBoxLayout()

        self.open_button = QPushButton("Abrir Imagem")
        self.open_button.clicked.connect(self.open_image)
        self.lateral_layout.addWidget(self.open_button)

        self.action_list = QListWidget()
        self.action_list.itemClicked.connect(self.apply_current_action) 
        self.lateral_layout.addWidget(self.action_list)

        for action in self.default_actions:
            self.action_list.addItem(action.nome)

        self.open_button = QPushButton("Aplicar Operação")
        self.open_button.clicked.connect(self.apply_action)
        self.lateral_layout.addWidget(self.open_button)

        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.remove_item)
        self.lateral_layout.addWidget(self.history_list)

        self.clear_button = QPushButton("Limpar Histórico")
        self.clear_button.clicked.connect(self.clear_history)
        self.lateral_layout.addWidget(self.clear_button)

        self.lateral_widget.setLayout(self.lateral_layout)
        main_layout.addWidget(self.lateral_widget)

        # Widget principal
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def apply_current_action(self, item):
        self.selected_action = item.text()

    def apply_action(self):
        if self.selected_image is not None:
            if self.selected_action not in self.selected_elements:
                self.selected_elements.append(self.selected_action)
                self.history_list.addItem(self.selected_action)
                self.actions()

    def remove_item(self, item=None):
        self.history_list.takeItem(self.history_list.row(item))
        self.selected_elements.remove(item.text())
        self.actions()

    def clear_history(self):
        self.history_list.clear()
        self.selected_elements.clear()
        self.actions()

    def actions(self):
        image_copy = self.selected_image.copy()

        for element in self.selected_elements:
            if element == "Converter para Cinza" and len(image_copy.shape) == 3:
                gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
                image_copy = gray_image
                 
            if element == "Filtro - Gaussian Blur":
                # Implemente o método de filtro aqui (por exemplo, filtro de suavização)
                # paramentros: imagem, tamanho do kernel, desvio padrão
                filtered_image = cv2.GaussianBlur(image_copy, (5, 5), 0)
                image_copy = filtered_image
                 
            if element == "Detectar Bordas - Canny":
                # Implemente o método de detector de borda aqui (por exemplo, Canny)
                # parametros: imagem, limiar minimo, limiar maximo
                edge_image = cv2.Canny(image_copy, 100, 200)
                image_copy = edge_image
               
            if element == "Binarizar Imagem":
                # Implemente o método de binarização aqui (por exemplo, limiar simples)
                _, binary_image = cv2.threshold(image_copy, 127, 255, cv2.THRESH_BINARY)
                image_copy = binary_image
              
            if element == "Morfologia Matemática - erosão":
                # Implemente o método de morfologia matemática aqui (por exemplo, erosão e dilatação)
                kernel = np.ones((5, 5), np.uint8)
                erosion_image = cv2.erode(image_copy, kernel, iterations=1)
                image_copy = erosion_image
              
        self.display_image(image_copy)

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