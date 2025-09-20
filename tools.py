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


def load_data():
    with open("rostro_catalogo.json", encoding="utf-8") as f:
        data = json.load(f)

    # Cargar tipos_rostro
    tipos_rostro = []
    for rostro in data.get("tipos_rostro", []):
        tipos_rostro.append(
            {
                "codigo": rostro.get("codigo"),
                "descripcion": rostro.get("descripcion"),
                "imagen": rostro.get("imagen"),
                "indices": rostro.get("indices", {}),
            }
        )
    df_rostros = pd.DataFrame(tipos_rostro)

    # Cargar partes (hombres y mujeres)
    partes = []
    for genero in ["hombres", "mujeres"]:
        for parte, items in data["partes"][genero].items():
            for item in items:
                partes.append(
                    {
                        "genero": genero,
                        "parte": parte,
                        "codigo": item.get("codigo"),
                        "descripcion": item.get("descripcion"),
                        "imagen": item.get("imagen"),
                        "indices": item.get("indices", {}),
                    }
                )
    df_partes = pd.DataFrame(partes)

    # Retornar ambos DataFrames en un diccionario
    return {"tipos_rostro": df_rostros, "partes": df_partes}


def load_catalog():
    catalog_df = pd.read_csv("catalog.csv")
    return catalog_df.parte.unique()


def redimenxiona_misma_forma(image_list):
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


def seleccionar_boca_ovalada(img, centro=None, tamaño_ovalo=None):
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


def bordes_boca_frecuencial(
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

    # Crear filtro de paso alto (máscara circular en el centro)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    # r = 10  # radio del área central (bajas frecuencias)
    mask[crow - r : crow + r, ccol - r : ccol + r] = 0

    # Aplicar filtro en el dominio frecuencial
    fshift_filtrado = fshift * mask

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
