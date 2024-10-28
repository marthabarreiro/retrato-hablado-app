import pandas as pd
import os
from typing import Tuple
import numpy as np
import cv2 as cv


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
    catalog_df = pd.read_csv("catalog.csv")
    faceparts_df = pd.read_csv("face_parts.csv")
    faceparts_df["codigo"] = faceparts_df["codigo"].astype(str).str.zfill(3)

    merged_df = pd.merge(
        catalog_df, faceparts_df, left_on="codigo", right_on="codigo_catalogo"
    )

    merged_df["image_name"] = merged_df["codigo_x"] + merged_df["codigo_y"]

    merged_df["image_path"] = merged_df["image_name"].apply(get_image_path)

    return merged_df


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
