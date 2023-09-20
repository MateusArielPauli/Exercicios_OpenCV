import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
)

import matplotlib.pyplot as plt
import cv2


class ImageConverterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_image = None
         
        self.setWindowTitle("Color Space Converter")
        self.setGeometry(100, 100, 200, 300)

        layout = QVBoxLayout()

        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.open_image)
        layout.addWidget(self.select_button)

        self.select_button1 = QPushButton("RGB -> GRAY")
        self.select_button1.clicked.connect(self.bgr2gray)
        layout.addWidget(self.select_button1)
        self.select_button1.setDisabled(True)
  
        self.select_button2 = QPushButton("RGB -> XYZ")
        self.select_button2.clicked.connect(self.bgr2xyz)
        layout.addWidget(self.select_button2)
        self.select_button2.setDisabled(True)
        
        self.select_button3 = QPushButton("RGB -> YCrCb")
        self.select_button3.clicked.connect(self.bgr2ycrcb)
        layout.addWidget(self.select_button3)
        self.select_button3.setDisabled(True)
        
        self.select_button4 = QPushButton("RGB -> HSV")
        self.select_button4.clicked.connect(self.bgr2hsv)
        layout.addWidget(self.select_button4)
        self.select_button4.setDisabled(True)
        
        self.select_button5 = QPushButton("RGB -> HLS")
        self.select_button5.clicked.connect(self.bgr2hls)
        layout.addWidget(self.select_button5)
        self.select_button5.setDisabled(True)        
        
        self.select_button6 = QPushButton("RGB -> CIE L*a*b*")
        self.select_button6.clicked.connect(self.bgr2lab)
        layout.addWidget(self.select_button6)
        self.select_button6.setDisabled(True)
                
        self.select_button7 = QPushButton("RGB -> CIE L*u*v*")
        self.select_button7.clicked.connect(self.bgr2luv)
        layout.addWidget(self.select_button7)
        self.select_button7.setDisabled(True)
        
        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)

    def bgr2gray(self):
        gray_image = cv2.cvtColor(self.selected_image, cv2.COLOR_BGR2GRAY)
        self.show_image("GRAY", gray_image, False)
       
    
    def bgr2xyz(self):
        xyz_image = cv2.cvtColor(self.selected_image, cv2.COLOR_BGR2XYZ)
        self.show_image("XYZ", xyz_image, True)
         
            
    def bgr2ycrcb(self):
        ycrcb_image = cv2.cvtColor(self.selected_image, cv2.COLOR_BGR2YCrCb)
        self.show_image("YCrCb", ycrcb_image,True)
     
            
    def bgr2hsv(self):
        hsv_image = cv2.cvtColor(self.selected_image, cv2.COLOR_BGR2HSV)
        self.show_image("HSV", hsv_image ,True) 
   
            
    def bgr2hls(self):
        hls_image = cv2.cvtColor(self.selected_image, cv2.COLOR_BGR2HLS)
        self.show_image("HLS", hls_image ,True)
         
            
    def bgr2lab(self):
        lab_image = cv2.cvtColor(self.selected_image, cv2.COLOR_BGR2Lab)
        self.show_image("CIE L*a*b*", lab_image ,True )
           
                    
    def bgr2luv(self):
        luv_image = cv2.cvtColor(self.selected_image, cv2.COLOR_BGR2Luv)
        self.show_image("CIE L*u*v*", luv_image ,True)
      
    def show_image(self, name, image, use_hsv_split=False):
        if use_hsv_split:
            h, s, v = cv2.split(image)
            cv2.imshow('Channel h', h)
            cv2.imshow('Channel s', s)
            cv2.imshow('Channel v', v)
            
            self.save_histogram(image, "Histogram")
            self.save_histogram(h, "Hue_Histogram")
            self.save_histogram(s, "Saturation_Histogram")
            self.save_histogram(v, "Value_Histogram")

        self.plot_normalized_histogram(image)   
        cv2.imshow(name, image)
         
    def plot_normalized_histogram(self, image):
        colors = ('b', 'g', 'r')

        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)  # Normaliza o histograma
            plt.plot(hist, color=color)
        
        plt.title('Normalized Image Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Normalized Frequency')
        plt.legend(['Blue Channel', 'Green Channel', 'Red Channel'])
        plt.show()
                
    def save_histogram(self, image, title):
        output_filename = f"./processed_{title}_image.png"
        cv2.imwrite(output_filename, image)
        print(f"Processed image saved as '{output_filename}'.") 
        
    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp *.gif);;All Files(*)", options=options)

        if image_path:
            try:
                self.selected_image = cv2.imread(image_path)
                if self.selected_image is not None:
                    cv2.imshow("RGB", self.selected_image)

                    for button in [self.select_button1, self.select_button2, self.select_button3, self.select_button4, self.select_button5, self.select_button6, self.select_button7]:
                        button.setDisabled(False)

                    cv2.waitKey(0)
                else:
                    print("Error of loading image.")
            except Exception as e:
                print("Error of opening image:", str(e))
        else:
            print("No image selected.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ImageConverterApp()
    main_window.show()
    sys.exit(app.exec_())
