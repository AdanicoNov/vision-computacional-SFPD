from PIL import Image
import math
import matplotlib.pyplot as plt
import numpy as np

class EstimuloVisual:
    def __init__(self):
        self.size = (512, 512)
        self.k1 = math.log(math.pi*511)/511
        self.k2 = 1/(self.k1*511)
        self.rawpixels = np.zeros(self.size)
        self.pixels = np.zeros(self.size)
        self.fi = 0
        self.min = 0
        self.max = 255

    def fiFunction(self, x):
        self.fi = self.k2 * math.exp(self.k1 * x)
        result = math.sin(self.fi)
        return result

    def g(self, y):
        # Función de atenuación
        return math.exp(-y*0.016)

    def normData(self):
        self.min = np.min(self.rawpixels)
        self.max = np.max(self.rawpixels)

    def generarImg(self, width, height):
        imgGen = Image.new('L', (width, height))  # 'L' para imagen en escala de grises

        for x in range(width):
            ranx = self.fiFunction(x)
            for y in range(height):
                attenuated_value = ranx * self.g(y)
                self.rawpixels[x, y] = attenuated_value
                self.pixels[x, y] = attenuated_value

        self.normData()

        # Normalizar y ajustar los valores de los píxeles
        for x in range(width):
            for y in range(height):
                normalized_value = (self.rawpixels[x, y] - self.min) / (self.max - self.min)
                pix = round(255 * normalized_value)
                imgGen.putpixel((x, -y), pix)

        return imgGen

# main
estimulo = EstimuloVisual()

img = estimulo.generarImg(512, 512)
plt.imshow(img, cmap='gray')
plt.show()

img.save("efectoVisual.jpg")
