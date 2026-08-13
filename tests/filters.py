"""
Filtros en dominio frecuencial.

Convencion:
  - El primer parametro de cada filtro es siempre `image: np.ndarray` (2D, escala de grises).
  - Los parametros siguientes son los controles que se mostraran en la GUI.
  - Type hints (int / float) determinan el tipo de widget generado.
  - Default values son los valores iniciales de los sliders.
  - El decorador @filter_params(param={'min':..., 'max':...}) define los rangos.
  - Cada filtro retorna la tupla (resultado_espacial, mascara_frecuencial).
"""

import numpy as np


def filter_params(**kwargs):
    """
    Decorador para especificar rango min/max de cada parametro en la GUI.

    Ejemplo:
        @filter_params(radius={'min': 1, 'max': 500})
        def low_pass(image: np.ndarray, radius: int = 30): ...
    """

    def decorator(func):
        func.__filter_params__ = kwargs
        return func

    return decorator


# ---------------------------------------------------------------------------
# Helpers privados (no se muestran en la GUI)
# ---------------------------------------------------------------------------


def _distance_map(shape: tuple) -> np.ndarray:
    """Mapa de distancias desde el centro de la imagen en el dominio frecuencial."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    return np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2)


def _apply_mask_fft(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Aplica una mascara en el dominio de la frecuencia y retorna la imagen espacial."""
    f = np.fft.fft2(image.astype(np.float64))
    fshift = np.fft.fftshift(f)
    filtered_shift = fshift * mask
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_shift)))
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Filtros publicos
# ---------------------------------------------------------------------------


@filter_params(radius={"min": 1, "max": 500})
def low_pass(image: np.ndarray, radius: int = 30):
    """Filtro pasa bajos: conserva frecuencias por debajo del radio."""
    dist = _distance_map(image.shape)
    mask = (dist <= radius).astype(np.float64)
    return _apply_mask_fft(image, mask), mask


@filter_params(radius={"min": 1, "max": 500})
def high_pass(image: np.ndarray, radius: int = 30):
    """Filtro pasa altos: conserva frecuencias por encima del radio."""
    dist = _distance_map(image.shape)
    mask = (dist > radius).astype(np.float64)
    return _apply_mask_fft(image, mask), mask


@filter_params(
    low_radius={"min": 1, "max": 498},
    high_radius={"min": 2, "max": 499},
)
def band_pass(image: np.ndarray, low_radius: int = 20, high_radius: int = 60):
    """Filtro pasa banda: conserva frecuencias entre low_radius y high_radius."""
    if low_radius >= high_radius:
        raise ValueError("low_radius debe ser menor que high_radius.")
    dist = _distance_map(image.shape)
    mask = ((dist >= low_radius) & (dist <= high_radius)).astype(np.float64)
    return _apply_mask_fft(image, mask), mask


@filter_params(
    low_radius={"min": 1, "max": 498},
    high_radius={"min": 2, "max": 499},
)
def band_reject(image: np.ndarray, low_radius: int = 20, high_radius: int = 60):
    """Filtro rechaza banda: elimina frecuencias entre low_radius y high_radius."""
    if low_radius >= high_radius:
        raise ValueError("low_radius debe ser menor que high_radius.")
    dist = _distance_map(image.shape)
    mask = ~((dist >= low_radius) & (dist <= high_radius))
    mask = mask.astype(np.float64)
    return _apply_mask_fft(image, mask), mask
