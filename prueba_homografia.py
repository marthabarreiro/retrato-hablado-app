import cv2
import numpy as np
import math


def seleccionar_puntos(nombre_ventana, imagen, diccionario_puntos):
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            etiqueta = input(f"{nombre_ventana} - Etiqueta para ({x},{y}): ")
            diccionario_puntos[etiqueta] = (x, y)
            cv2.circle(imagen, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(nombre_ventana, imagen)

    cv2.imshow(nombre_ventana, imagen)
    cv2.setMouseCallback(nombre_ventana, click)
    print(
        f"Selecciona puntos con etiquetas en {nombre_ventana}. Presiona ESC para terminar."
    )
    while True:
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyWindow(nombre_ventana)


def distancia(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Cargar imágenes
rostro = cv2.imread("images/plantillas/tp001_pc.fw.png")
componente = cv2.imread("images/menton/mm001.jpg")

rostro_copy = rostro.copy()

componente_copy = componente.copy()

# Diccionarios de puntos
puntos_rostro = {}
puntos_componente = {}

# Seleccionar puntos en ambas imágenes
seleccionar_puntos("Rostro", rostro_copy, puntos_rostro)
seleccionar_puntos("Componente", componente_copy, puntos_componente)

# Encontrar etiquetas comunes
etiquetas = list(set(puntos_rostro.keys()) & set(puntos_componente.keys()))
if len(etiquetas) < 2:
    raise ValueError(
        "Se requieren al menos 2 puntos coincidentes con la misma etiqueta en ambas imágenes."
    )

# Calcular escalado promedio basado en distancias entre pares de puntos
dist_rostro = []
dist_componente = []
for i in range(len(etiquetas)):
    for j in range(i + 1, len(etiquetas)):
        pi_r = puntos_rostro[etiquetas[i]]
        pj_r = puntos_rostro[etiquetas[j]]
        pi_c = puntos_componente[etiquetas[i]]
        pj_c = puntos_componente[etiquetas[j]]
        dist_rostro.append(distancia(pi_r, pj_r))
        dist_componente.append(distancia(pi_c, pj_c))

factor_escala = np.mean(np.array(dist_rostro) / np.array(dist_componente))
print(f"Factor de escala medio: {factor_escala:.2f}")

# Redimensionar componente
componente_escalado = cv2.resize(
    componente, None, fx=factor_escala, fy=factor_escala, interpolation=cv2.INTER_LINEAR
)

# Calcular traslaciones por cada punto, luego promediar
traslaciones = []
for etiqueta in etiquetas:
    px_r = np.array(puntos_rostro[etiqueta])
    px_c = np.array(puntos_componente[etiqueta]) * factor_escala
    t = px_r - px_c
    traslaciones.append(t)

traslacion_media = np.mean(traslaciones, axis=0).astype(int)
print(f"Traslación media: {traslacion_media}")

# Aplicar la transformación
output = rostro.copy()
h, w = rostro.shape[:2]
comp_h, comp_w = componente_escalado.shape[:2]

for y in range(comp_h):
    for x in range(comp_w):
        xi = x + traslacion_media[0]
        yi = y + traslacion_media[1]
        if 0 <= xi < w and 0 <= yi < h:
            pixel = componente_escalado[y, x]
            if not np.all(pixel == 0):  # evitar fondo negro
                output[yi, xi] = pixel

cv2.imshow("Resultado", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
