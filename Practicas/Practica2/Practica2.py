import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from PIL import Image
class dithering():
    def __init__(self, imgSize, routeImg):
        self.route = routeImg
        self.image= cv2.imread(routeImg)
        self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        self.normImage = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX)
        self.imgHalftone = np.zeros(imgSize)
        self.imgRandDithering = np.zeros(imgSize)
        self.imgOrdDithering = np.zeros(imgSize)
        self.imgRandDithering = np.zeros(imgSize)
        self.size=imgSize
        self.matriz2 = np.array([[1,3],[2,0]])
        self.matriz4 = np.array([[15,7,13,5],[3,11,1,9],[12,4,14,6],[0,8,2,10]])

    def cuantizad1Bit(self):
        # Load the image
        image = Image.open(self.route)
        # Convert the image to grayscale
        image = image.convert('L')
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
        # Save the quantized image
        image.save('cuantizada.png')
        return image

    def orderDithering(self, offset):

        if (offset==2):
            matriz_umbral = self.matriz2
        else:
            matriz_umbral = self.matriz4

        imagen = self.normImage
        filas, columnas = self.normImage.shape
        tam_img = matriz_umbral.shape[0]
        self.imgHalftone = np.zeros_like(self.normImage)

        for i in range(filas):
            for j in range(columnas):
                # índice de la matriz de umbral
                x = i % tam_img
                y = j % tam_img
                umbral = (matriz_umbral[x, y] + 0.5) * (254 / (tam_img * tam_img))

                # Aplicar el umbral
                if imagen[i, j] > umbral:
                    self.imgHalftone[i, j] = 254  # Blanco
                else:
                    self.imgHalftone[i, j] = 0    # Negro
        orderDithering = Image.fromarray(self.imgHalftone)
        orderDithering.save('orderDithering.png')

    def random_dithering(self):
        image = Image.fromarray(self.normImage)
        image = image.convert('L')
        width, height = image.size
        pixels = np.array(image)

        # Creamos una matriz de las misma dimensiones que la matriz original
        dithering_matrix = np.random.randint(0, 256, (height, width), dtype=np.uint8)

        # Aplicamos dithering
        dithered_pixels = np.where(pixels < dithering_matrix + 0.5, 0, 255).astype(np.uint8)

        # Creamos la imagen a partir de los píxeles de dithering
        dithered_image = Image.fromarray(dithered_pixels)
        dithImg = Image.fromarray(dithered_pixels, mode='L')
        dithImg.save('randomDithering.png')
        return dithered_image


    def floyd_steinberg(self):
        image=self.normImage
        h, w = image.shape
        # Doble iteración sobre altura (h) y anchura (w)
        for y in range(h):
            for x in range(w):
                old = image[y, x]
                reajuste = np.round(old)
                image[y, x] = reajuste
                error = old - reajuste
                # Reajuste de valores en función del error y los valores adyacentes
                if x + 1 < w:
                    image[y, x + 1] += error * 0.4375 # derecha, 7 / 16
                if (y + 1 < h) and (x + 1 < w):
                    image[y + 1, x + 1] += error * 0.0625 # derecha_abajo, 1 / 16
                if y + 1 < h:
                    image[y + 1, x] += error * 0.3125 # abajo, 5 / 16
                if (x - 1 >= 0) and (y + 1 < h):
                    image[y + 1, x - 1] += error * 0.1875 # izquierda_abajo, 3 / 16
        image = Image.fromarray(image)
        image.save('floyd_steinberg.png')


dither = dithering((200,200),'./visual.jpeg')
cuantizar = dither.cuantizad1Bit()
dither.random_dithering()
dither.orderDithering(2)
dither.orderDithering(4)
dither.floyd_steinberg()

