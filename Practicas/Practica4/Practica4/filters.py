import numpy as np
import cv2
from itertools import permutations
import math
class filters:
    def __init__(self):
        pass
        
    

    def salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
       
        noisy_image = np.copy(image)
        
        # Añadir ruido de sal (píxeles blancos)
        num_salt = np.ceil(salt_prob * image.size)
        coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[coords_salt[0], coords_salt[1]] = 1  # Para imágenes en escala de grises

        # Añadir ruido de pimienta (píxeles negros)
        num_pepper = np.ceil(pepper_prob * image.size)
        coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[coords_pepper[0], coords_pepper[1]] = 0  # Para imágenes en escala de grises

        return noisy_image
    
    def gauss_vector(self, l):
        """
        Args:
            l (int): level in the gauss pyramid
        """
        gauss_1d = np.zeros(l+1)
        for x in range(l+1):
            gauss_1d[x] = math.factorial(l)/(math.factorial(x)*math.factorial(l-x))
        return gauss_1d

    def gauss_matrix(self,l):
        gauss_1d = filters.gauss_vector(self, l)
        filter_gauss = np.outer(gauss_1d, np.transpose(gauss_1d))
        scalar = np.sum(filter_gauss)
        return filter_gauss/scalar
    
    def to_matrix(self, vector1, vector2):
        filter_mat = np.outer(vector1, vector2)
        scalar = np.sum(filter_mat)
        return filter_mat
    
    def derv1_gauss(self,l):
        gauss_d = filters.gauss_vector(self, l)
        gauss_1d = np.zeros((l+1))
        gauss_1d = np.convolve(gauss_d,(1,-1))
        #print(gauss_1d)
        return gauss_1d
        
    def derv2_gauss(self,l):
        gauss_1d = filters.derv1_gauss(self,l)
        gauss_2d = np.zeros((l+1))
        gauss_2d = np.convolve(gauss_1d,(1,-1))
        #print(gauss_2d)
        return gauss_2d
    
    def low_step_block(self, n):
        k_block = np.ones((n,n), dtype=int)/(n*n)
        return k_block
    
    def convolution_2d(image, kernel):
        row1, col1 = image.shape[0], image.shape[0]
        row2, col2 = kernel.shape[0],kernel.shape[0]
        result = [[0]*(col1 + col2 -1) for _ in range(row1+row2 -1)]
        for i in range(row1):
            for j in range(col1):
                for k in range(row2):
                    for l in range(col2):
                        result[i + k][j + l] += image[i][j] * kernel[k][l]
        
        return result
if __name__==  "__main__":
    filtros = filters()
    k_gauss = filtros.gauss_vector(3)
    #print(k_gauss)
    k_gauss = filtros.gauss_matrix(3)
    print(k_gauss)
    #k_dervgauss= filtros.derv2_gauss(3)
    #print(k_dervgauss)
    
    image = cv2.imread('artecontemp.png', cv2.IMREAD_GRAYSCALE)
    print(image.shape[0])