import numpy as np
import cv2
import matplotlib.pyplot as plt


def calcular_matriz_homografia(puntos_origen, puntos_destino):
    # Convertimos los puntos a numpy arrays
    puntos_origen = np.array(puntos_origen, dtype=np.float32)
    puntos_destino = np.array(puntos_destino, dtype=np.float32)
    
    matriz_homografia = cv2.getPerspectiveTransform(puntos_origen, puntos_destino)
    
    return matriz_homografia

def aplicar_transformacion_homografia(imagen, matriz_homografia, tamaño_salida):
    # Aplicar la transformación de perspectiva a la imagen
    imagen_transformada = cv2.warpPerspective(imagen, matriz_homografia, tamaño_salida)
    return imagen_transformada


def verificar_transformacion(imagen_original, puntos_origen, puntos_destino, matriz_afin, tamaño_salida):
    # Aplicar la transformación
    imagen_transformada = aplicar_transformacion_homografia(imagen_original, matriz_afin, tamaño_salida)

    # Dibujar los puntos de origen en la imagen original
    imagen_original_puntos = imagen_original.copy()
    for punto in puntos_origen:
        cv2.circle(imagen_original_puntos, punto, 5, (0, 0, 255), -1)  # Puntos en rojo

    # Dibujar los puntos de destino en la imagen transformada
    imagen_transformada_puntos = imagen_transformada.copy()
    for punto in puntos_destino:
        cv2.circle(imagen_transformada_puntos, punto, 5, (0, 255, 0), -1)  # Puntos en verde

    # Mostrar ambas imágenes (original con puntos y transformada con puntos)
    plt.figure(figsize=(10, 5))

    # Imagen original con puntos
    plt.subplot(1, 2, 1)
    plt.title('Imagen Original con Puntos')
    plt.imshow(cv2.cvtColor(imagen_original_puntos, cv2.COLOR_BGR2RGB))

    # Imagen transformada con puntos
    plt.subplot(1, 2, 2)
    plt.title('Imagen Transformada con Puntos')
    plt.imshow(cv2.cvtColor(imagen_transformada_puntos, cv2.COLOR_BGR2RGB))

    plt.show()
import cv2
img = cv2.imread("./Ajedrez_Proyectado.jpg")
tamaño_salida = (918,814)
# Definir los puntos de referencia
#v5
puntos_origen = [(46, 368), (613, 661), (430, 153), (885, 290)]  # Puntos en la imagen original
puntos_destino = [(116, 57), (111, 752), (803, 64), (804, 749)]  # Puntos a los que queremos mapear

matriz_homografia = calcular_matriz_homografia(puntos_origen, puntos_destino)
print("Matriz de Homografía:")
print(matriz_homografia)
# Aplicar la transformación de perspectiva a la imagen
imagen_transformada = aplicar_transformacion_homografia(img, matriz_homografia, tamaño_salida)
plt.imshow(cv2.cvtColor(imagen_transformada, cv2.COLOR_BGR2RGB))
plt.show()

verificar_transformacion(img, puntos_origen, puntos_destino, matriz_homografia, tamaño_salida)
# Guardar la imagen transformada

cv2.imwrite('imagen_transformada_afin.jpg', imagen_transformada)
