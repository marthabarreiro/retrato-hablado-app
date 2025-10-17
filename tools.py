import pandas as pd
import os
from typing import Tuple
import numpy as np
import cv2 as cv
import json


def get_image_path(image_name):
    base_image_dir = "images/"
    for directory in os.listdir(base_image_dir):
        image_path = os.path.join(base_image_dir, directory, f"{image_name}.jpg")
        if os.path.exists(image_path):
            return image_path
        else:
            image_path = os.path.join(base_image_dir, directory, f"{image_name}.png")
            if os.path.exists(image_path):
                return image_path
    return None


def read_file():
    with open("rostro_catalogo.json", encoding="utf-8") as f:
        data = json.load(f)

    # Lista para almacenar todos los registros
    todos_registros = []

    # Cargar tipos_rostro con etiqueta específica
    for rostro in data.get("tipo_rostro", []):
        todos_registros.append(
            {
                "tipo": "tipo_rostro",  # Identificador del tipo de elemento
                "genero": "unisex",  # Los tipos de rostro son unisex
                "parte": "tipo_rostro",
                "codigo": rostro.get("codigo"),
                "descripcion": rostro.get("descripcion"),
                "imagen": rostro.get("imagen"),
                "indices": rostro.get("indices", {}),
            }
        )

    # Cargar partes (hombres y mujeres)
    for genero in ["hombres", "mujeres"]:
        for parte, items in data["partes"][genero].items():
            for item in items:
                todos_registros.append(
                    {
                        "tipo": "parte_facial",  # Identificador del tipo de elemento
                        "genero": genero,
                        "parte": parte,
                        "codigo": item.get("codigo"),
                        "descripcion": item.get("descripcion"),
                        "imagen": item.get("imagen"),
                        "indices": item.get("indices", {}),
                    }
                )

    # Crear un solo DataFrame con todos los registros
    df_unificado = pd.DataFrame(todos_registros)
    return df_unificado


def redimenxiona_misma_forma(image_list):
    image_list = [cv.imread(img_path, cv.IMREAD_GRAYSCALE) for img_path in image_list]
    # Encontrar las dimensiones máximas (alto y ancho) de las imágenes
    max_height = max(img.shape[0] for img in image_list)
    max_width = max(img.shape[1] for img in image_list)

    # Crear una lista para almacenar las imágenes con relleno
    padded_images = []

    for img in image_list:
        # Obtener el alto y ancho de la imagen actual
        height, width = img.shape[:2]

        # Calcular los márgenes de relleno en alto y ancho
        top_padding = (max_height - height) // 2
        bottom_padding = max_height - height - top_padding
        left_padding = (max_width - width) // 2
        right_padding = max_width - width - left_padding

        # Crear la imagen con relleno utilizando np.pad
        if img.ndim == 3:  # Si la imagen es RGB (3 canales)
            padded_img = np.pad(
                img,
                ((top_padding, bottom_padding), (left_padding, right_padding), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:  # Si la imagen es en escala de grises (1 canal)
            padded_img = np.pad(
                img,
                ((top_padding, bottom_padding), (left_padding, right_padding)),
                mode="constant",
                constant_values=0,
            )

        # Añadir la imagen con relleno a la lista
        padded_images.append(padded_img)

    return padded_images


def recortar_ovalo(img, centro=None, tamaño_ovalo=None):
    # Cargar imagen
    if img is None:
        raise ValueError("No se pudo cargar la imagen.")

    h, w = img.shape[:2]

    # Definir centro y tamaño del óvalo si no se especifican
    if centro is None:
        centro = (w // 2, h // 2)
    if tamaño_ovalo is None:
        tamaño_ovalo = (int(w * 0.45), int(h * 0.45))  # ancho, alto

    # Crear máscara con óvalo blanco sobre fondo negro
    mask = np.zeros((h, w), dtype=np.uint8)
    cv.ellipse(mask, centro, tamaño_ovalo, 0, 0, 360, 255, -1)

    # Aplicar máscara: fuera del óvalo se pone en blanco
    img_blanco = np.ones_like(img) * 255
    if img.ndim == 3:
        resultado = np.where(mask[..., None] == 255, img, img_blanco)
    else:
        resultado = np.where(mask == 255, img, img_blanco)

    return resultado


def bordes_frecuencial(
    img,
    r=10,
    umbral=10,
):
    """
    Obtiene los bordes de la boca aplicando un filtro de paso alto en el dominio frecuencial.
    Algoritmos matemáticos: FFT 2D, filtro de paso alto, IFFT 2D.
    Args:
        img (np.ndarray): Imagen de la boca.
        r (int): Radio del área central (bajas frecuencias) a eliminar.
        umbral (int): Umbral para la binarización de la imagen.
    Returns:
        np.ndarray: Imagen con los bordes resaltados.
    """
    if img is None:
        raise ValueError("No se pudo cargar la imagen.")

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = normalizar_grises(img)
    # Transformada de Fourier 2D
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Crear filtro de paso alto gaussiano (máscara gaussiana en el centro)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Crear coordenadas para la máscara gaussiana
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    xx, yy = np.meshgrid(x, y)

    # Calcular distancia desde el centro
    distance = np.sqrt((xx - ccol) ** 2 + (yy - crow) ** 2)

    # Crear máscara gaussiana invertida (paso alto)
    # sigma controla el ancho de la gaussiana (relacionado con r)
    sigma = r / 2.0
    gaussian_mask = 1 - np.exp(-(distance**2) / (2 * sigma**2))

    # Aplicar filtro en el dominio frecuencial
    fshift_filtrado = fshift * gaussian_mask

    # Transformada inversa para regresar al dominio espacial
    f_ishift = np.fft.ifftshift(fshift_filtrado)
    img_bordes = np.fft.ifft2(f_ishift)
    img_bordes = np.abs(img_bordes)
    _, img_bordes_bin = cv.threshold(
        img_bordes,
        umbral,
        255,
        cv.THRESH_BINARY_INV,
    )
    return img_bordes_bin


def normalizar_grises(img):
    """
    Normaliza una imagen en escala de grises para que sus valores estén entre 0 y 255.
    Args:
        img (np.ndarray): Imagen en escala de grises.
    Returns:
        np.ndarray: Imagen normalizada.
    """
    if img is None:
        raise ValueError("Imagen no válida")
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.uint8)
    img_norm = (img - min_val) / (max_val - min_val) * 255
    return img_norm.astype(np.uint8)


def show_image(image, title="Image"):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def filtro_dodge_burn(imagen_gris):
    """
    Efecto de boceto usando técnicas de dodge y burn.

    Este método simula el proceso fotográfico tradicional de dodge (aclarar)
    y burn (oscurecer) para crear un efecto de dibujo a lápiz realista.

    Proceso:
    1. Inversión de la imagen para obtener un "negativo"
    2. Desenfoque del negativo para crear una capa de textura suave
    3. Dodge blending: división de color que realza los contrastes

    Args:
        imagen_gris: Imagen en escala de grises (np.ndarray)

    Returns:
        np.ndarray: Imagen con efecto de boceto artístico
    """
    # 1. Invertir la imagen (negativo fotográfico)
    # Convierte píxeles oscuros en claros y viceversa
    # Fórmula: pixel_invertido = 255 - pixel_original
    imagen_invertida = cv.bitwise_not(imagen_gris)
    # 2. Aplicar desenfoque gaussiano a la imagen invertida
    # El kernel (21, 21) crea un desenfoque suave que preserva estructuras
    # grandes mientras elimina detalles finos
    # Esto actúa como una "capa de iluminación difusa"
    imagen_blur = cv.GaussianBlur(imagen_invertida, (21, 21), 0)

    # 3. Aplicar la operación de "dodge" (aclarado fotográfico)
    # Esta técnica simula exponer selectivamente áreas de la imagen
    def dodge(front, back):
        """
        Dodge blend mode (modo de mezcla aclarar).

        Fórmula: resultado = (front * 256) / (256 - back)

        Donde:
        - front: imagen original (capa frontal)
        - back: imagen desenfocada invertida (capa de fondo)
        - 256: escala para evitar saturación

        El resultado realza los bordes y crea el efecto de dibujo
        porque las áreas claras se vuelven más brillantes mientras
        que las áreas oscuras mantienen contraste.
        """
        # cv.divide con scale=256 implementa: (front * 256) / (256 - back)
        # Evita división por cero y normaliza automáticamente a 0-255
        result = cv.divide(front, 255 - back, scale=256)
        return result

    # Aplicar dodge blending entre la imagen original y el blur invertido
    # Esto crea el efecto final de boceto al lápiz
    boceto = dodge(imagen_gris, imagen_blur)
    boceto = recortar_ovalo(boceto)
    return boceto


def flip_image(image: np.ndarray) -> np.ndarray:
    return cv.flip(image, 0)


def suma_imagenes_dominio_frecuencial(imgs: list):
    # Asegurarse de que todas las imágenes tengan el mismo tamaño
    imgs_redimensionadas = redimenxiona_misma_forma(imgs)

    # Inicializar la suma en el dominio frecuencial
    suma_frecuencial = None

    for img in imgs_redimensionadas:
        # Aquí se aplica el filtro
        img = filtro_dodge_burn(img)
        # Transformada de Fourier 2D
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        if suma_frecuencial is None:
            suma_frecuencial = fshift
        else:
            suma_frecuencial += fshift

    # Transformada inversa para regresar al dominio espacial
    f_ishift = np.fft.ifftshift(suma_frecuencial)
    img_suma = np.fft.ifft2(f_ishift)
    img_suma = np.abs(img_suma)

    # Normalizar la imagen resultante a 0-255
    img_suma_norm = normalizar_grises(img_suma)

    return img_suma_norm
