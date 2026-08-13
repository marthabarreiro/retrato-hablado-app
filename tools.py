import json
import os
import numpy as np
import cv2 as cv
import pandas as pd


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


def aplica_mascara_cuadrada_suavizada(img, padding):
    """
    Aplica una máscara cuadrada con bordes suavizados que gradualmente
    llegan a 255 (blanco) y suma el resultado a la imagen original.

    Args:
        img (np.ndarray): Imagen de entrada
        padding (int): Tamaño del borde en píxeles para la transición

    Returns:
        np.ndarray: Imagen con bordes suavizados sumados
    """
    if img is None:
        raise ValueError("No se pudo cargar la imagen.")

    h, w = img.shape[:2]

    # Crear máscara inversa: 0 en el centro, aumenta gradualmente hacia los bordes
    # Esta máscara contendrá los valores que se sumarán (blanco en bordes)
    mask = np.zeros((h, w), dtype=np.float32)

    # Crear una máscara auxiliar con 255 en los bordes
    mask_bordes = np.ones((h, w), dtype=np.float32) * 255
    mask_bordes[padding : h - padding, padding : w - padding] = 0

    # Suavizar los bordes de la máscara usando un filtro gaussiano
    # Esto crea una transición gradual de 0 a 255
    kernel_size = 2 * padding + 1 if padding > 0 else 3
    mask_suavizada = cv.GaussianBlur(mask_bordes, (kernel_size, kernel_size), 0)

    # Convertir imagen a float para operaciones
    img_float = img.astype(np.float32)

    # Sumar la máscara suavizada a la imagen original
    if img.ndim == 3:
        # Para imágenes a color, expandir la máscara
        resultado = img_float + mask_suavizada[..., None]
    else:
        # Para escala de grises
        resultado = img_float + mask_suavizada

    # Asegurar que los valores estén en el rango 0-255
    resultado = np.clip(resultado, 0, 255).astype(img.dtype)

    return resultado


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


def filtro_dodge(imagen_gris):
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
    inverted_image = cv.bitwise_not(imagen_gris)
    imagen_blur = cv.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)
    img = imagen_gris.astype(np.float32)
    blur = imagen_blur.astype(np.float32)
    epsilon = 1e-6
    sketch = cv.divide(img, 255.0 - blur + epsilon, scale=256.0)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)
    return sketch


def flip_image(image: np.ndarray) -> np.ndarray:
    return cv.flip(image, 0)


def save_image(image: np.ndarray, output_dir: str = "output") -> str:
    """
    Guarda una imagen en el directorio especificado con nombre basado en timestamp.

    Args:
        image: Imagen a guardar (np.ndarray)
        output_dir: Directorio donde guardar la imagen (por defecto "output")

    Returns:
        str: Ruta completa donde se guardó la imagen
    """
    import datetime

    # Crear directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generar nombre único con timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"retrato_hablado_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Guardar imagen
    cv.imwrite(filepath, image)

    return filepath


def suma_imagenes_dominio_frecuencial(parts_data: list):
    """
    Suma imágenes de partes faciales en el dominio frecuencial.

    Args:
        parts_data: Lista de diccionarios con estructura:
                   [{'imagen': 'path/to/image.png', 'indices': {...}, ...}, ...]
                   El primer elemento debe ser la plantilla base.

    Returns:
        np.ndarray: Imagen resultante normalizada (escala de grises)
    """
    if not parts_data or len(parts_data) < 1:
        raise ValueError("Se requiere al menos una imagen (plantilla base)")

    # Cargar la plantilla base (primer elemento)
    plantilla = {
        "imagen": cv.imread(parts_data[0]["imagen"], cv.IMREAD_GRAYSCALE),
        "indices": parts_data[0].get("indices", {}),
    }

    # Ajustar el tono de piel de las partes faciales al de la plantilla
    only_imgs_list = [cv.imread(part_data["imagen"]) for part_data in parts_data[1:]]
    imgs_list_same_skin = []
    if len(only_imgs_list) >= 4:
        imgs_list_same_skin = match_skin_tone(only_imgs_list, ref_index=3)

    # Cargar las partes faciales (resto de elementos)
    imgs_list = []
    for idx, part_data in enumerate(parts_data[1:]):
        if part_data.get("parte") != "orejas":
            imgs_list.append(
                {
                    "imagen": filtro_dodge(
                        cv.imread(part_data["imagen"], cv.IMREAD_GRAYSCALE)
                    ),
                    "parte": part_data.get("parte", ""),
                    "indices": part_data.get("indices", {}),
                    "original_idx": idx,
                }
            )
        else:
            # Agregar la imagen original de la oreja
            imagen_oreja = cv.imread(part_data["imagen"], cv.IMREAD_GRAYSCALE)
            imgs_list.append(
                {
                    "imagen": filtro_dodge(imagen_oreja),
                    "parte": "orejas",
                    "indices": part_data.get("indices", {}),
                    "original_idx": idx,
                }
            )
            # Agregar la imagen con flip horizontal
            imgs_list.append(
                {
                    "imagen": filtro_dodge(cv.flip(imagen_oreja, 1)),
                    "parte": "orejas_flip",
                    "indices": part_data.get("indices", {}),
                    "original_idx": idx,
                }
            )
    # Inicializar la suma en el dominio frecuencial
    suma_frecuencial = None
    suma_same_skin = None

    for i in range(len(imgs_list)):
        img = colocar_parte_en_plantilla(
            plantilla["imagen"],
            aplica_mascara_cuadrada_suavizada(imgs_list[i]["imagen"], padding=5),
            plantilla["indices"].get(imgs_list[i]["parte"], {}),
            imgs_list[i]["indices"],
        )

        # Manejar la versión con tono de piel ajustado
        original_idx = imgs_list[i].get("original_idx")
        if original_idx is not None and original_idx < len(imgs_list_same_skin):
            try:
                skin_img = imgs_list_same_skin[original_idx]
                # Si es la oreja con flip, aplicar el flip también a la imagen con piel
                if imgs_list[i]["parte"] == "orejas_flip":
                    skin_img = cv.flip(skin_img, 1)
                
                img_same_skin = colocar_parte_en_plantilla(
                    cv.cvtColor(plantilla["imagen"], cv.COLOR_GRAY2BGR),
                    aplica_mascara_cuadrada_suavizada(skin_img, padding=5),
                    plantilla["indices"].get(imgs_list[i]["parte"], {}),
                    imgs_list[i]["indices"],
                )
                
                if suma_same_skin is None:
                    suma_same_skin = img_same_skin
                else:
                    # Solo rellenar píxeles vacíos (blancos), no sumar en intersecciones
                    mask_vacio = np.all(suma_same_skin > 250, axis=2)
                    suma_same_skin[mask_vacio] = img_same_skin[mask_vacio]
                    
            except Exception as e:
                print(f"Error ajustando tono de piel: {e}")

        # Transformada de Fourier 2D
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        if suma_frecuencial is None:
            suma_frecuencial = fshift
        else:
            suma_frecuencial += fshift

    # Se muestra imagen con tono ajustado solo para comprobar, esto es código muerto y puede ser eliminado si no se necesita
    # if suma_same_skin is not None and len(only_imgs_list) >= 4:
    #     show_image(
    #         suma_same_skin.astype(np.uint8), title="Rostro con tono de piel ajustado"
    #     )
    # Transformada inversa para regresar al dominio espacial
    f_ishift = np.fft.ifftshift(suma_frecuencial)
    img_suma = np.fft.ifft2(f_ishift)
    img_suma = np.abs(img_suma)

    # Normalizar la imagen resultante a 0-255
    boceto_final = normalizar_grises(img_suma)
    # Normalizar la imagen con tono de piel ajustado a 0-255 si existe
    boceto_recortes = suma_same_skin.astype(np.uint8) if suma_same_skin is not None else None

    return boceto_final, boceto_recortes


def colocar_parte_en_plantilla(
    plantilla, parte_rostro, indices_plantilla, indices_parte
):
    """
    Coloca una parte del rostro en su posición correcta sobre una plantilla base.

    Utiliza el algoritmo de escalado y traslación basado en puntos de referencia
    comunes entre la plantilla y la parte del rostro.

    Args:
        plantilla (np.ndarray): Imagen de la plantilla base (fondo blanco)
        parte_rostro (np.ndarray): Imagen de la parte del rostro a colocar
        indices_plantilla (dict): Diccionario con puntos de referencia de la plantilla
                                  Estructura: {"nombre_punto": {"x": int, "y": int}, ...}
        indices_parte (dict): Diccionario con puntos de referencia de la parte
                              Debe tener las mismas claves que indices_plantilla

    Returns:
        np.ndarray: Imagen resultante con las dimensiones de la plantilla y la parte colocada

    Ejemplo de uso:
        indices_plantilla = {
            "sad": {"x": 411, "y": 206},
            "zyd": {"x": 395, "y": 277},
            "sbad": {"x": 391, "y": 341}
        }
        indices_parte = {
            "sad": {"x": 46, "y": 8},
            "zyd": {"x": 13, "y": 85},
            "sbad": {"x": 9, "y": 139}
        }
        resultado = colocar_parte_en_plantilla(plantilla, oreja, indices_plantilla, indices_parte)
    """
    # Crear copia de la plantilla para no modificar el original
    output = np.ones(plantilla.shape, dtype=plantilla.dtype) * 255

    # Encontrar etiquetas comunes entre plantilla y parte
    etiquetas_comunes = list(set(indices_plantilla.keys()) & set(indices_parte.keys()))

    if len(etiquetas_comunes) < 2:
        raise ValueError(
            f"Se requieren al menos 2 puntos coincidentes. "
            f"Encontrados: {len(etiquetas_comunes)} ({etiquetas_comunes})"
        )

    # Función auxiliar para calcular distancia euclidiana
    def distancia(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # Convertir diccionarios de índices a tuplas (x, y)
    puntos_plantilla = {k: (v["x"], v["y"]) for k, v in indices_plantilla.items()}
    puntos_parte = {k: (v["x"], v["y"]) for k, v in indices_parte.items()}

    # Calcular factor de escala promedio basado en distancias entre pares de puntos
    dist_plantilla = []
    dist_parte = []

    for i in range(len(etiquetas_comunes)):
        for j in range(i + 1, len(etiquetas_comunes)):
            etiq_i = etiquetas_comunes[i]
            etiq_j = etiquetas_comunes[j]

            # Distancias en plantilla
            pi_plantilla = puntos_plantilla[etiq_i]
            pj_plantilla = puntos_plantilla[etiq_j]
            dist_plantilla.append(distancia(pi_plantilla, pj_plantilla))

            # Distancias en parte
            pi_parte = puntos_parte[etiq_i]
            pj_parte = puntos_parte[etiq_j]
            dist_parte.append(distancia(pi_parte, pj_parte))

    # Calcular factor de escala medio
    factor_escala = np.mean(np.array(dist_plantilla) / np.array(dist_parte))

    # Redimensionar la parte del rostro
    parte_escalada = cv.resize(
        parte_rostro,
        None,
        fx=factor_escala,
        fy=factor_escala,
        interpolation=cv.INTER_LINEAR,
    )

    # Calcular traslación promedio para alinear los puntos
    traslaciones = []
    for etiqueta in etiquetas_comunes:
        px_plantilla = np.array(puntos_plantilla[etiqueta])
        px_parte = np.array(puntos_parte[etiqueta]) * factor_escala
        traslacion = px_plantilla - px_parte
        traslaciones.append(traslacion)

    traslacion_media = np.mean(traslaciones, axis=0).astype(int)

    # Obtener dimensiones
    h_plantilla, w_plantilla = plantilla.shape[:2]
    h_parte, w_parte = parte_escalada.shape[:2]

    # Aplicar la transformación: colocar la parte escalada en la plantilla
    for y in range(h_parte):
        for x in range(w_parte):
            # Calcular posición en la plantilla
            xi = x + traslacion_media[0]
            yi = y + traslacion_media[1]

            # Verificar que esté dentro de los límites
            if 0 <= xi < w_plantilla and 0 <= yi < h_plantilla:
                pixel = parte_escalada[y, x]

                # Solo copiar píxeles no blancos (evitar fondo blanco)
                # Umbral de 250 para considerar "blanco"
                if len(pixel.shape) == 0:  # Escala de grises
                    if pixel < 250:
                        output[yi, xi] = pixel
                else:  # Color
                    if not np.all(pixel > 250):
                        output[yi, xi] = pixel

    return output


def skin_mask(img):
    # img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    # mask = cv.inRange(img_ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    # return cv.GaussianBlur(mask, (7, 7), 0)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    H, S, V = cv.split(img_hsv)
    Y, Cr, Cb = cv.split(img_ycrcb)

    # Rango piel en HSV
    mask1 = cv.inRange(img_hsv, (0, 40, 0), (25, 255, 255))
    mask2 = cv.inRange(img_hsv, (165, 40, 0), (180, 255, 255))  # tonos rojos extremos

    hsv_mask = cv.bitwise_or(mask1, mask2)

    # Rango piel en YCrCb (mucho más preciso)
    ycrcb_mask = cv.inRange(img_ycrcb, (0, 135, 85), (255, 180, 135))

    # Combinar ambas máscaras
    mask = cv.bitwise_and(hsv_mask, ycrcb_mask)

    # Suavizar bordes
    mask = cv.medianBlur(mask, 7)
    mask = cv.GaussianBlur(mask, (9, 9), 0)

    # ELIMINAR OJOS Y LABIOS (zonas oscuras brillantes)
    # Crear una máscara inversa basada en luminancia
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    eyes_mouth = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)[1]
    eyes_mouth = cv.GaussianBlur(eyes_mouth, (17, 17), 0)

    # Quitar ojos/dientes
    mask = cv.bitwise_and(mask, cv.bitwise_not(eyes_mouth))

    return mask


def reinhard_color_transfer(src, target, mask_src, mask_tgt):
    """
    Transfers color characteristics from target to source using Reinhard method.
        
    Matches mean and standard deviation of color channels in LAB color space
    to harmonize skin tones between different facial parts.
    
    Args:
        src: Source image to adjust
        target: Target image to match
        mask_src: Mask defining regions to adjust in source
        mask_tgt: Mask defining reference regions in target
        
    Returns:
        Source image with adjusted colors
    """
    src_lab = cv.cvtColor(src, cv.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv.cvtColor(target, cv.COLOR_BGR2LAB).astype(np.float32)
    Ls, As, Bs = cv.split(src_lab)
    Lt, At, Bt = cv.split(tgt_lab)
    As_pix = As[mask_src > 0]
    Bs_pix = Bs[mask_src > 0]
    At_pix = At[mask_tgt > 0]
    Bt_pix = Bt[mask_tgt > 0]
    if len(As_pix) == 0 or len(At_pix) == 0:
        return src
    mean_As, std_As = As_pix.mean(), max(As_pix.std(), 1)
    mean_Bs, std_Bs = Bs_pix.mean(), max(Bs_pix.std(), 1)
    mean_At, std_At = At_pix.mean(), At_pix.std()
    mean_Bt, std_Bt = Bt_pix.mean(), Bt_pix.std()
    A_new = As.copy()
    B_new = Bs.copy()
    A_new[mask_src > 0] = ((As_pix - mean_As) * (std_At / std_As)) + mean_At
    B_new[mask_src > 0] = ((Bs_pix - mean_Bs) * (std_Bt / std_Bs)) + mean_Bt
    result_lab = cv.merge([Ls, A_new, B_new])
    return cv.cvtColor(result_lab.astype(np.uint8), cv.COLOR_LAB2BGR)


def match_skin_tone(images, ref_index=0):
    """
    Adjusts skin tone across a list of images to match a reference.
    
    All images will have their skin tones adjusted to match the reference
    image specified by ref_index, creating visual harmony across facial parts.
    
    Args:
        images: List of BGR images to process
        ref_index: Index of the reference image (default: 0)
        
    Returns:
        List of images with adjusted skin tones
    """
    reference = images[ref_index]
    mask_ref = skin_mask(reference)
    adjusted = []
    for i, img in enumerate(images):
        if i == ref_index:
            adjusted.append(img)
            continue
        mask_src = skin_mask(img)
        result = reinhard_color_transfer(img, reference, mask_src, mask_ref)
        result = cv.bilateralFilter(result, 7, 50, 50)
        adjusted.append(result)
    return adjusted
