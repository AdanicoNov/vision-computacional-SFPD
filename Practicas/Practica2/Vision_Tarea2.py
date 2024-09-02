from PIL import Image
import numpy as np

# Cargamos la imagen
image = Image.open('BorisPicture.jpeg')
#image = Image.open('papa_bañando_abuelo.jpeg')
# Convertimos la imagen a escala de grises
image = image.convert('L')

# Hacemos la cuantización de 1 Bit
image_quantized = image.point(lambda x: 0 if x < 128 else 255, '1')


#Guardamos la imagen para el caso A)
image_quantized.save('1_bit_quantized.png')
#-------------------------------------------Metodo de Dithering Aleatorio---------------------------------------------#
def random_dithering(image):
    width, height = image.size
    pixels = np.array(image)

    # Creamos una matriz de las misma dimensiones que la matriz original
    dithering_matrix = np.random.randint(0, 256, (height, width), dtype=np.uint8)

    # Aplicamos dithering
    dithered_pixels = np.where(pixels < dithering_matrix, 0, 255).astype(np.uint8)

    # Creamos la imagen a partir de los píxeles de dithering
    dithered_image = Image.fromarray(dithered_pixels, mode='L')
    return dithered_image


#Guardamos la imagen para el caso B), usando el método de dithering aleatorio
image_random_dithering = random_dithering(image)
#Guardamos la de dithering
image_random_dithering.save('random_dithering.png')

#Probamos el dithering + la cuantizacion
test = random_dithering(image_quantized)
test.save('test_img.png')

