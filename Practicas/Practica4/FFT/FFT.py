import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import filtros as fl
import cv2
class FFT:
    def __init__(self):
        pass
    # Función para generar los coeficientes binomiales para el filtro
    def binomial_coefficients(n):
        coeffs = [1]
        for i in range(n):
            coeffs.append(coeffs[-1] * (n - i) // (i + 1))
        return np.array(coeffs)

    # Función para generar el kernel 2D binomial gaussiano
    def binomial_kernel_2d(self,n):
        kernel_1d = self.binomial_coefficients(n)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / np.sum(kernel_2d)  # Normalizamos el kernel
        return kernel_2d

    # Función para aplicar el padding al filtro
    def pad_filter(self, filtro, img_shape):
        filtro_padded = np.zeros(img_shape)
        f_h, f_w = filtro.shape   # Alto y ancho del filtro
        img_h, img_w = img_shape  # Alto y ancho de la imagen
        filtro_padded[:f_h, :f_w] = filtro
        return filtro_padded

    def bin_paso_bajas(self,img, n, img_name):
        kernel = fl.filters.gauss_matrix(self,n)
        blurred_image = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(f'./3/3_binomial_{n}_{img_name}.png',blurred_image)
        return kernel, blurred_image

if __name__ == "__main__":
    fft = FFT()
    #Cargamos la imagen 
    image = cv2.imread('original.png', cv2.IMREAD_GRAYSCALE)

    #Creamos el kernel 
    kernel, imageBlurTeam= fft.bin_paso_bajas(image,11,'FFT')

    #Obtenemos el tamaño de la imagen 
    img_shape = image.shape
    
    #Hacemos el padding con zeros alrededor del kernel 
    padded_kernel = fft.pad_filter(kernel, img_shape)


    fft_image = np.fft.fft2(image)
    fft_kernel = np.fft.fft2(padded_kernel)
    fft_kernel_shifted  = np.fft.fftshift(fft_kernel)
    #Este es el que vamos a graficar
    magnitudeSpectrum = np.log(np.abs(fft_kernel_shifted) + 1) 


    fft_result = fft_image * fft_kernel
    fft_result_shiftted = np.fft.fftshift(fft_result)
    resultSpectrum = np.log(np.log(np.abs(fft_result_shiftted) + 1))


    convolved_image = np.fft.ifft2(fft_result)
    
    convolved_image = np.real(convolved_image)

    #plt.figure(figsize=(4, 4))
    #plt.imshow(kernel, cmap='gray')
    #plt.axis('off')
    #plt.show()


    plt.imshow(convolved_image, cmap='gray')
    plt.axis('off')
    plt.show()
