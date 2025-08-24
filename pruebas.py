from tools import seleccionar_boca_ovalada, show_image, bordes_boca_frecuencial
import cv2 as cv
import numpy as np

ruta_boca = "images/boca/mb003.png"
boca_original = cv.imread(ruta_boca)
show_image(boca_original, "Boca Original")
boca_filtrada = bordes_boca_frecuencial(boca_original, r=2, umbral=40)
show_image(boca_filtrada, "Boca Filtrada")
boca_recortada = seleccionar_boca_ovalada(boca_filtrada)
show_image(boca_recortada, "Boca Ovalada")
