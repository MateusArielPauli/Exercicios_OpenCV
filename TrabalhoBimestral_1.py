import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
)
from PyQt5.QtWidgets import QSlider
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class PdiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_image = None
        self.modified_image = None  # Imagem modificada (para manter o estado atual)

        self.setWindowTitle("Trabalho Bimestral")
        self.setGeometry(100, 100, 300, 300)

        self.blur_intensity = 15  # Adicionando o atributo blur_intensity

        # Layout principal
        main_layout = QVBoxLayout()

        # Botões para ações
        convert_button = QPushButton("Converção de cor: GRAY", self)
        convert_button.clicked.connect(self.apply_conversion_GRAY)
        main_layout.addWidget(convert_button)
        convert_button.setStyleSheet("font-size: 14px")  # Tamanho fonte

        convert_button2 = QPushButton("Converção de cor: HLS", self)
        convert_button2.clicked.connect(self.apply_conversion_HLS)
        main_layout.addWidget(convert_button2)
        convert_button2.setStyleSheet("font-size: 14px")  # Tamanho fonte

        main_layout.addSpacing(50)
        
        median_button = QPushButton("Filtro: Mediana", self)
        median_button.clicked.connect(self.apply_median_filter)
        main_layout.addWidget(median_button)
        median_button.setStyleSheet("font-size: 14px")  # Tamanho fonte

        # Slider para controlar a intensidade do filtro blur
        blur_slider = QSlider(Qt.Horizontal, self)
        blur_slider.setMinimum(1)  # Valor mínimo do slider
        blur_slider.setMaximum(30)  # Valor máximo do slider
        blur_slider.setValue(15)  # Valor inicial do slider
        blur_slider.valueChanged.connect(self.update_blur_intensity)
        main_layout.addWidget(blur_slider)
        # Criar um label para indicar a intensidade do filtro de mediana
        self.blur_intensity_label = QLabel('Blur Intensity: 15', self)
        main_layout.addWidget(self.blur_intensity_label)

        main_layout.addSpacing(50)

        laplace_button = QPushButton("Detector de Bordas: Laplace", self)
        laplace_button.clicked.connect(self.apply_laplace_edge_detection)
        main_layout.addWidget(laplace_button)
        laplace_button.setStyleSheet("font-size: 14px")  # Tamanho fonte

        # Slider para controlar o tamanho do kernel (para o detector de bordas de Laplace)
        self.kernel_slider = QSlider(Qt.Horizontal, self)
        self.kernel_slider.setMinimum(1)  # Valor mínimo do slider
        self.kernel_slider.setMaximum(30)  # Valor máximo do slider
        self.kernel_slider.setValue(3)  # Valor inicial do slider
        self.kernel_slider.valueChanged.connect(self.apply_laplace_edge_detection)
        main_layout.addWidget(self.kernel_slider)
        # Criar um label para indicar o valor do slider (tamanho do kernel)
        self.kernel_value_label = QLabel('Kernel Size: 3', self)
        main_layout.addWidget(self.kernel_value_label)

        main_layout.addSpacing(50)

        binary_button = QPushButton("Binarizar imagem: Threshould", self)
        binary_button.clicked.connect(self.apply_binarization)
        main_layout.addWidget(binary_button)
        binary_button.setStyleSheet("font-size: 14px")  # Tamanho fonte

        # Slider para controlar o valor de threshould para a binarização
        self.thresh_slider = QSlider(Qt.Horizontal, self)
        self.thresh_slider.setMinimum(0)  # Valor mínimo do slider
        self.thresh_slider.setMaximum(255)  # Valor máximo do slider
        self.thresh_slider.setValue(127)  # Valor inicial do slider
        self.thresh_slider.valueChanged.connect(self.apply_binarization)
        main_layout.addWidget(self.thresh_slider)
        # Criar um label para indicar o valor do slider (valor de threshould)
        self.thresh_value_label = QLabel('Threshold Value: 127', self)
        main_layout.addWidget(self.thresh_value_label)

        main_layout.addSpacing(50)

        erosion_button = QPushButton("Morfologia Matemática: Erosão", self)
        erosion_button.clicked.connect(self.apply_erosion)
        main_layout.addWidget(erosion_button)
        erosion_button.setStyleSheet("font-size: 14px")  # Tamanho fonte

        # Slider para controlar o tamanho do kernel para a operação de erosão
        self.erosion_kernel_slider = QSlider(Qt.Horizontal, self)
        self.erosion_kernel_slider.setMinimum(1)  # Valor mínimo do slider
        self.erosion_kernel_slider.setMaximum(20)  # Valor máximo do slider
        self.erosion_kernel_slider.setValue(5)  # Valor inicial do slider
        self.erosion_kernel_slider.valueChanged.connect(self.apply_erosion)
        main_layout.addWidget(self.erosion_kernel_slider)
        # Criar um label para indicar o valor do slider (tamanho do kernel para a erosão)
        self.erosion_kernel_value_label = QLabel('Erosion Kernel Size: 5', self)
        main_layout.addWidget(self.erosion_kernel_value_label)

        main_layout.addSpacing(20)

        # Slider para controlar o número de iterações para a operação de erosão
        self.erosion_slider = QSlider(Qt.Horizontal, self)
        self.erosion_slider.setMinimum(1)  # Valor mínimo do slider
        self.erosion_slider.setMaximum(10)  # Valor máximo do slider
        self.erosion_slider.setValue(1)  # Valor inicial do slider
        self.erosion_slider.valueChanged.connect(self.apply_erosion)
        main_layout.addWidget(self.erosion_slider)
        # Criar um label para indicar o valor do slider (número de iterações para a erosão)
        self.erosion_value_label = QLabel('Erosion Iterations: 1', self)
        main_layout.addWidget(self.erosion_value_label)

        # Adicionar espaçamento entre os botões
        main_layout.addSpacing(80)

        open_button = QPushButton("Abrir Imagem", self)
        open_button.clicked.connect(self.open_image)
        main_layout.addWidget(open_button)
        open_button.setStyleSheet("font-size: 14px")  # Tamanho fonte

        reset_button = QPushButton("Reverter para Original", self)
        reset_button.clicked.connect(self.reset_image)
        main_layout.addWidget(reset_button)
        reset_button.setStyleSheet("font-size: 14px")  # Tamanho fonte

        #Salvar a nova imagem
        save_button = QPushButton("Salvar Imagem Editada", self)
        save_button.clicked.connect(self.save_image)
        main_layout.addWidget(save_button)
        save_button.setStyleSheet("font-size: 14px")  # Tamanho fonte
    
        # Label para exibir a imagem
        self.image_label = QLabel(self)
        main_layout.addWidget(self.image_label)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def update_blur_intensity(self, value):
        # Garantir que o valor seja ímpar
        self.blur_intensity = value if value % 2 == 1 else value + 1
        # Atualizar o valor da intensidade no label
        self.blur_intensity_label.setText(f'Blur Intensity: {self.blur_intensity}')

        if self.modified_image is not None:
            # Aplicar o filtro de mediana
            filtered_image = cv2.medianBlur(self.modified_image, self.blur_intensity)
            self.display_image(filtered_image)
        
        #self.modified_image = filtered_image
        return filtered_image
        
    def update_kernel_size(self, value):
        # Garantir que o valor seja ímpar (necessário para o kernel do detector de bordas de Laplace)
        kernel_size = value if value % 2 == 1 else value + 1
        self.kernel_size = kernel_size
        self.actions()  # Aplicar a ação do detector de bordas de Laplace em tempo real

    def apply_conversion_GRAY(self):
        if self.modified_image is not None and len(self.modified_image.shape) == 3 and self.modified_image.shape[2] == 3:
            gray_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2GRAY)
            self.display_image(gray_image)
            self.modified_image = gray_image
        else:
            print("A imagem não é colorida (3 canais).")
        return self.modified_image

    def apply_conversion_HLS(self):
        if self.modified_image is not None and len(self.modified_image.shape) == 3 and self.modified_image.shape[2] == 3:
            hls_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2HLS)
            self.display_image(hls_image)
            self.modified_image = hls_image
        else:
            print("A imagem não é colorida (3 canais).")
        return self.modified_image


    def apply_median_filter(self):
        if self.modified_image is not None:
            filtered_image = cv2.medianBlur(self.modified_image, self.blur_intensity)
            self.display_image(filtered_image)
            self.modified_image = filtered_image
        return self.modified_image

    def apply_laplace_edge_detection(self):
        if self.modified_image is not None and len(self.modified_image.shape) == 3 and self.modified_image.shape[2] == 3:
            # Obter o tamanho do kernel (garantir que seja ímpar)
            kernel_size = self.kernel_slider.value()
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size - 1

            # Aplicar o detector de bordas de Laplace
            gray_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=kernel_size)
            edges = cv2.convertScaleAbs(edges)
            self.display_image(edges)
            # Atualizar o valor do kernel no label
            self.kernel_value_label.setText(f'Kernel Size: {self.kernel_slider.value()}')
        else:
            # Obter o tamanho do kernel (garantir que seja ímpar)
            kernel_size = self.kernel_slider.value()
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size - 1

            # Aplicar o detector de bordas de Laplace
            edges = cv2.Laplacian(self.modified_image, cv2.CV_64F, ksize=kernel_size)
            edges = cv2.convertScaleAbs(edges)
            self.display_image(edges)
            # Atualizar o valor do kernel no label
            self.kernel_value_label.setText(f'Kernel Size: {self.kernel_slider.value()}')

    
    def apply_binarization(self):
        if self.modified_image is not None and len(self.modified_image.shape) == 3 and self.modified_image.shape[2] == 3:
            # Aplicar binarização com um limiar (threshold)
            gray_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, self.thresh_slider.value(), 255, cv2.THRESH_BINARY)
            self.display_image(binary_image)
            # Atualizar o valor de threshould no label
            self.thresh_value_label.setText(f'Threshold Value: {self.thresh_slider.value()}')
        else:
            # Aplicar binarização com um limiar (threshold)
            _, binary_image = cv2.threshold(self.modified_image, self.thresh_slider.value(), 255, cv2.THRESH_BINARY)
            self.display_image(binary_image)
            # Atualizar o valor de threshould no label
            self.thresh_value_label.setText(f'Threshold Value: {self.thresh_slider.value()}')


    def apply_erosion(self):
        if self.modified_image is not None:
            # Obter o tamanho do kernel para a erosão
            kernel_size = self.erosion_kernel_slider.value()

            # Aplicar operação de erosão
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            erosion_image = cv2.erode(self.modified_image, kernel, iterations=self.erosion_slider.value())
            self.display_image(erosion_image)
            # Atualizar o tamanho do kernel no label
            self.erosion_kernel_value_label.setText(f'Erosion Kernel Size: {kernel_size}')
            # Atualizar o número de iterações no label
            self.erosion_value_label.setText(f'Erosion Iterations: {self.erosion_slider.value()}')

    def reset_image(self):
        if self.selected_image is not None:
            self.modified_image = self.selected_image
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
                self.modified_image = self.selected_image
                self.display_image(self.modified_image)
                

            except Exception as e:
                print("Erro ao abrir a imagem:", str(e))
        else:
            print("Nenhuma iselected_image")
        

    def save_image(self):
        if self.modified_image is not None:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_path, _ = QFileDialog.getSaveFileName(self, "Salvar Imagem", "", "Images (*.png *.jpg *.bmp);;All Files(*)", options=options)
            if file_path:
                try:
                    cv2.imwrite(file_path, self.modified_image)
                    print("Imagem salva com sucesso.")
                except Exception as e:
                    print("Erro ao salvar a imagem:", str(e))
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = PdiApp()
    main_window.show()
    sys.exit(app.exec_())