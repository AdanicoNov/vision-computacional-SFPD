import numpy as np
import cv2
import filters as fl
class appliend_filters:
    def __init__(self, image):
        self.pepper_noise_img = fl.filters.salt_pepper_noise(image,0.018)
        
        
    def paso_bajas(self, img, n, img_name):
        kernel = fl.filters.low_step_block(self,n)
        blurred_image = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(f'./2/2_pasobajas_{n}_{img_name}.png',blurred_image)
        return blurred_image
    
    def bin_paso_bajas(self, img, n, img_name):
        kernel = fl.filters.gauss_matrix(self,n)
        blurred_image = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(f'./3/3_binomial_{n}_{img_name}.png',blurred_image)
        return blurred_image
    
    def edge_block(self, img, img_name):
        kernel = np.array([-1,1])
        #print(kernel)
        #edge_detec = np.convolve(img.reshape(img.shape(0)*img.shape(1)),1,(1,-1))
        edge_detec = cv2.filter2D(img, -1, kernel)
        edge_detec_numpy = np.array(edge_detec)
        #img_normalized = cv2.normalize(edge_detec_numpy, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'./4/4a_block_{img_name}.png',edge_detec_numpy)
        #print(edge_detec)
    
    def Prewitt_x(self, img, img_name):
        low_step = np.ones(5)
        kernel = fl.filters.derv1_gauss(self,3)
        kernel =  fl.filters.to_matrix(self,np.transpose(low_step),np.flip(kernel))
        print(kernel)
        edge_detec = fl.filters.convolution_2d(img,kernel)
        edge_detec_numpy = np.array(edge_detec)
        img_normalized = cv2.normalize(edge_detec_numpy, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'./4/4a_binomial_prewitt_{img_name}.png',img_normalized)
        #print(img_normalized)
        
    def Prewitt_y(self, img, img_name):
        
        low_step = np.ones(5)
        kernel = fl.filters.derv1_gauss(self,3)
        kernel =  fl.filters.to_matrix(self,np.transpose(np.flip(kernel)),low_step)
        print(kernel)
        
        edge_detec = fl.filters.convolution_2d(img,kernel)
        edge_detec_numpy = np.array(edge_detec)
        img_normalized = cv2.normalize(edge_detec_numpy, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'./4/4a_binomial_prewitt_{img_name}.png',img_normalized)
        
    
    def Sobel(self, img, img_name, flag):
        kernel_bin = fl.filters.gauss_vector(self, 4)
        kernel_1derv = fl.filters.derv1_gauss(self, 3)
        if flag: #En x
            kernel1 = np.transpose(kernel_bin)
            kernel2 = np.flip(kernel_1derv)
        else: 
            kernel1 = np.transpose(np.flip(kernel_1derv))
            kernel2 = kernel_bin
            
        kernel =  fl.filters.to_matrix(self,kernel1,kernel2)
        print(kernel)
        edge_detec = fl.filters.convolution_2d(img,kernel)
        edge_detec_numpy = np.array(edge_detec)
        img_normalized = cv2.normalize(edge_detec_numpy, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'./4/4c_binomial_sobel_{img_name}.png',img_normalized)
        #print(img_normalized)
        
    def dervGauss(self, n, img, img_name):
        kernel = fl.filters.derv1_gauss(self, n-2)
        print(kernel)
        kernel_mat = fl.filters.to_matrix(self, np.transpose(np.flip(kernel)), np.flip(kernel))
        edge_detec = fl.filters.convolution_2d(img,kernel_mat)
        edge_detec_numpy = np.array(edge_detec)
        img_normalized = cv2.normalize(edge_detec_numpy, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'./4/4d_deriv_gauss_{img_name}_b_{n}.png',img_normalized)
        #print(img_normalized)
        
    def laplace(self, img, img_name):
        kernel = fl.filters.laplace
        edge_detec = fl.filters.convolution_2d(img,kernel)
        edge_detec_numpy = np.array(edge_detec)
        img_normalized = cv2.normalize(edge_detec_numpy, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'./5/5a_laplace_{img_name}.png',img_normalized)
    
        
    def derv2Gauss(self, n, img, img_name):
        
        kernel = fl.filters.derv2_gauss(self, n-3)
        kernel_mat = fl.filters.to_matrix(self, np.transpose(np.flip(kernel)), np.flip(kernel))
        #kernel_mat = fl.filters.to_matrix(self,kernel,np.transpose(kernel))
        edge_detec = fl.filters.convolution_2d(img,kernel_mat)
        edge_detec_numpy = np.array(edge_detec)
        
        img_normalized = cv2.normalize(edge_detec_numpy, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'./5/5b_2deriv_gauss_{img_name}_{n}.png',img_normalized)
        #print(img_normalized)
        
    def unsharp_paso_bajas(self, n , img, img_name, k):
        imagecopy = np.copy(img)
        kernel = fl.filters.low_step_block(self,5)
        kernel_h = fl.filters.low_step_block(self,n)
        blurred_image = cv2.filter2D(img, -1, kernel)
        #identidad del impulso?
        identity = np.zeros((n*n)).reshape(n,n)
        identity[n//2][n//2] = 1 + k
        print(identity)
        #Filtro suavizador, bloque
        filterhl = identity + k*(identity -1/(n*n)*kernel_h) 
        
        edge_detec = fl.filters.convolution_2d(img,filterhl)
        edge_detec_numpy = np.array(edge_detec)
        
        img_normalized =edge_detec_numpy
        #img_normalized = cv2.normalize(edge_detec_numpy, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'./6/6a_unsharp_paso_bajas_{img_name}_{n}.png',img_normalized)
        print(filterhl)
        
    def unsharp_bin_paso_bajas(self, n , img, img_name, k):
        imagecopy = np.copy(img)
        kernel = fl.filters.low_step_block(self,5)
        kernel_gauss = fl.filters.gauss_matrix(self,n-1)
        blurred_image = cv2.filter2D(img, -1, kernel)
        #identidad del impulso?
        identity = np.zeros((n*n)).reshape(n,n)
        identity[n//2][n//2] = 1 + k
        print(identity)
        #Filtro suavizador, bloque
        filterhl = identity + k*(identity -1/(n*n)*kernel_gauss) 
        
        edge_detec = fl.filters.convolution_2d(img,filterhl)
        edge_detec_numpy = np.array(edge_detec)
        
        img_normalized =edge_detec_numpy
        #img_normalized = cv2.normalize(edge_detec_numpy, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'./6/6b_unsharp_bin_paso_bajas_{img_name}_{n}.png',img_normalized)
        print(filterhl)
        
        
        
    
if __name__==  "__main__":
    #Load original image
    image = cv2.imread('artecontemp.png', cv2.IMREAD_GRAYSCALE)
    filters_applied = appliend_filters(image)
    #imagen with noise
    peper_img = filters_applied.pepper_noise_img
    
    #cv2.imwrite(f'orign.png',image)
    #cv2.imwrite(f'noise.png',peper_img)
    
    #Filters applied 3x3, 7x7, 9x9, 11x11
    """ 
    blurred_image = filters_applied.paso_bajas(image, 3, 'normal')
    blurred_image_noise = filters_applied.paso_bajas(peper_img, 3, 'noise')
    
    blurred_image = filters_applied.paso_bajas(image, 7, 'normal')
    blurred_image_noise = filters_applied.paso_bajas(peper_img, 7, 'noise')
    
    blurred_image = filters_applied.paso_bajas(image, 9, 'normal')
    blurred_image_noise = filters_applied.paso_bajas(peper_img, 9, 'noise')
    
    blurred_image = filters_applied.paso_bajas(image, 11, 'normal')
    blurred_image_noise = filters_applied.paso_bajas(peper_img, 11, 'noise') """
    
    #Binomial Filter 3x3, 7x7, 9x9, 11x11
    """ blurred_img = filters_applied.bin_paso_bajas(image, 3, 'normal')
    blurred_img_noise = filters_applied.bin_paso_bajas(peper_img, 3, 'noise')
    
    blurred_img = filters_applied.bin_paso_bajas(image, 7, 'normal')
    blurred_img_noise = filters_applied.bin_paso_bajas(peper_img, 7, 'noise')
    
    blurred_img = filters_applied.bin_paso_bajas(image, 9, 'normal')
    blurred_img_noise = filters_applied.bin_paso_bajas(peper_img, 9, 'noise')
    
    blurred_img = filters_applied.bin_paso_bajas(image, 11, 'normal')
    blurred_img_noise = filters_applied.bin_paso_bajas(peper_img, 11, 'noise') """
     
    """ filters_applied.edge_block(image, 'normal_sinnorm')
    filters_applied.edge_block(peper_img, 'noise_sinnorm') """
    """ edge_img = filters_applied.Prewitt_y(image, 'normal_y')
    edge_img = filters_applied.Prewitt_y(peper_img, 'noise_y')
    edge_img = filters_applied.Prewitt_x(image, 'normal_x')
    edge_img = filters_applied.Prewitt_x(peper_img, 'noise_x') """
    
    """edge_img = filters_applied.Sobel(image, 'normal_x', 1)
    edge_img = filters_applied.Sobel(image, 'normal_y', 0)
    edge_img = filters_applied.Sobel(peper_img, 'noise_x', 1)
    edge_img = filters_applied.Sobel(peper_img, 'noise_y', 0) """
    """
    filters_applied.unsharp_paso_bajas(3,image, 'normal', 0.98)
    filters_applied.unsharp_paso_bajas(3,peper_img, 'noise', 0.98)
    filters_applied.unsharp_bin_paso_bajas(3, image, 'normal', 0.98)
    filters_applied.unsharp_bin_paso_bajas(3, peper_img, 'noise', 0.98)
    filters_applied.unsharp_paso_bajas(7,image, 'normal', 0.98)
    filters_applied.unsharp_paso_bajas(7,peper_img, 'noise', 0.98)
    filters_applied.unsharp_bin_paso_bajas(7, image, 'normal', 0.98)
    filters_applied.unsharp_bin_paso_bajas(7, peper_img, 'noise', 0.98)
    
    filters_applied.dervGauss(5,image,'normal')
    filters_applied.dervGauss(5,peper_img,'noise') 
    
    filters_applied.dervGauss(7,image,'normal')
    filters_applied.dervGauss(7,peper_img,'noise')
    filters_applied.dervGauss(11,image,'normal')
    filters_applied.dervGauss(11,peper_img,'noise')
    """""""""
    
    #filters_applied.laplace(image, 'normal' )
    #filters_applied.laplace(peper_img, 'noise' )
    
    filters_applied.derv2Gauss(5,image,'normal')
    filters_applied.derv2Gauss(5,peper_img,'noise')
    filters_applied.derv2Gauss(7,image,'normal')
    filters_applied.derv2Gauss(7,peper_img,'noise')
    filters_applied.derv2Gauss(11,image,'normal')
    filters_applied.derv2Gauss(11,peper_img,'noise')
    """"""
    
    
    
    