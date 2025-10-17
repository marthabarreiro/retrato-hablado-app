from tools import seleccionar_ovalo, show_image, bordes_frecuencial
import cv2 as cv

ruta_boca = "images/boca/mb002.jpg"
boca_original = cv.imread(ruta_boca)
show_image(boca_original, "Boca Original")
boca_filtrada = bordes_frecuencial(boca_original, r=2, umbral=40)
show_image(boca_filtrada, "Boca Filtrada")
boca_recortada = seleccionar_ovalo(boca_filtrada)
show_image(boca_recortada, "Boca Ovalada")
