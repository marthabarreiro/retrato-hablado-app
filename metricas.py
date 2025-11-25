# metricas_dodge_burn.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io, color, filters, feature, img_as_float
import matplotlib.pyplot as plt

# --- 1) Lista EXACTA de tus 50 archivos (IDs = nombres de archivo) ---
# Nota: Los archivos están en subdirectorios de images/, no en la raíz
filenames = [
    # paq1
    "h_cua_p.lac_c.arq_o.estr_n.abul_b.gra.lab.del_m.cua.jpg",
    "h_cua_p.lac_c.rec_o.peq_n.resp_b.med.lab.del_m.sob.jpg",
    "h_cua_p.lac_c.sobres_o.asia_n.cha_b.peq.lab.del_m.red.jpg",
    "h_cua_p.red_c.tria_o.alm_n.agui_b.gra.lab.gru_m.redu.jpg",
    "h_cua_p.riz_c.rot_o.prom_n.resp_b.peq.lab.gru_m.alar.jpg",
    "h_ov_p.lac_c_arq_o.prom_n.cha_b.gra.lab.del_m.red.jpg",
    "h_ov_p.lac_c_rec_o.peq_n.agui_b.peq.lab.del_m.alar.jpg",
    "h_ov_p.riz_c_rot_o.estr_n.resp_b.peq.lab.gru_m.sob.jpg",
    "h_ov_p.riz_c_sobres_o.asia_n.abul_b.med.lab.gru_m.cua.jpg",
    "h_ov_p.riz_c_tri_o.alm_n.rec_b.gra.lab.gru_m.redu.jpg",
    # paq2
    "h_rec_p.lac_c.rot_o.peq_n.abul_b.gra.lab.del_m.sob.jpg",
    "h_rec_p.lac_c.sobres_o.estr_n.rec_b.peq.lab.del_m.cua.jpg",
    "h_rec_p.lac_c.tria_o.asi_n.cha_b.med.lab.del_m.redu.jpg",
    "h_rec_p.riz_c.arq_o.alm_n.agui_b.peq.lab.gru_m.red.jpg",
    "h_rec_p.riz_c.rec_o.prom_n.resp_b.med.lab.gru_m.alar.jpg",
    "h_red_p.lac_c.arq_o.peq_n.agui_b.peq.lab.del_m_alar.jpg",
    "h_red_p.lac_c.rec_o.peq_n.resp_b.med.lab.del_m_red.jpg",
    "h_red_p.riz_c.rot_o.peq_n.abul_b.med.lab.gru_m_cua.jpg",
    "h_red_p.riz_c.sobres_o.peq_n.rec_b.gra.lab.gru_m_sob.jpg",
    "h_red_p.riz_c.tri_o.peq_n.cha_b.peq.lab.gru_m_redu.jpg",
    # paq3
    "h_tri_p.lac_c.arq_o.asia_n.cha_b.gra.lab.del_m.cua.jpg",
    "h_tri_p.lacio_c.rot_o.peq_n.abul_b.peq.lab.del_m.red.jpg",
    "h_tri_p.lacio_c.tri_o.estr_n.rec_b.med.lab.del_m.redu.jpg",
    "h_tri_p.riz_c.rec_o.alm_n.agui_b.med.lab.gru_m.sob.jpg",
    "h_tri_p.riz_c.sobres_o.prom_n.resp_b.gra.lab.gru_m.alar.jpg",
    "m_cua_p.lac_c.arq_o.estr_n.abul_b.gra.lab.del_m.cua.jpg",
    "m_cua_p.lac_c.rec_o.peq_n.rec_b.med.lab.del_m.sob.jpg",
    "m_cua_p.lac_c.sobres_o.asia_n.cha_b.peq.lab.del_m.red.jpg",
    "m_cua_p.riz_c.rot_o.prom_n.res_b.peq.lab.gru_m.alar.jpg",
    "m_cua_p.riz_c.tria_o.alm_n.agui_b.gra.lab.gru_m.redu.jpg",
    # paq4
    "m_ov_p.lac_c.arq_o.prom_n.cha_b.gra.lab.del_m.red.jpg",
    "m_ov_p.lac_c.rec_o.peq_n.agui_b.peq.lab.del_m.alar.jpg",
    "m_ov_p.riz_c.rot_o.estr_n.resp_b.peq.lab.gru_m.sob.jpg",
    "m_ov_p.riz_c.sobres_o.asia_n.abul_b.med.lab.gru_m.cua.jpg",
    "m_ov_p.riz_c.tria_o.alm_n.rec_b.gra.lab.gru_m.redu.jpg",
    "m_rec_p.lac_c.rot_o.peq-n.abul_b.gra.lab.del_m.sob.jpg",
    "m_rec_p.lac_c.sobres_o.estr-n.rec_b.peq.lab.del_m.cua.jpg",
    "m_rec_p.lac_c.tria_o.asia-n.cha_b.med.lab.del_m.redu.jpg",
    "m_rec_p.riz_c.arq_o.alm-n.agui_b.peq.lab.gru_m.red.jpg",
    "m_rec_p.riz_c.rec_o.prom-n.resp_b.med.lab.gru_m.alar.jpg",
    # paq5
    "m_red_p.lac_c.arq_o.peq_n.agui_b.peq.lab.del_m.alar.jpg",
    "m_red_p.lac_c.rec_o.peq_n.resp_b.med.lab.del_m.red.jpg",
    "m_red_p.riz_c.rot_o.peq_n.abul_b.med.lab.gru_m.cua.jpg",
    "m_red_p.riz_c.sobres_o.peq_n.rec_b.gra.lab.gru_m.sob.jpg",
    "m_red_p.riz_c.tria_o.peq_n.cha_b.peq.lab.gru_m.redu.jpg",
    "m_tria_p.lac_c.arq_o.asia_n.cha_b.gra.lab.del_m.cua.jpg",
    "m_tria_p.lac_c.rot_o.peq_n.abul_b.peq.lab.del_m.red.jpg",
    "m_tria_p.lac_c.tria_o.estr_n.rec_b.med.lab.del_m.redu.jpg",
    "m_tria_p.riz_c.rec_o.alm_n.agui_b.med.lab.gru_m.sob.jpg",
    "m_tria_p.riz_c.sob_o.prom_n.resp_b.gra.lab.gru_m.alar.jpg",
]


# Importar función para buscar las rutas reales
# Función para buscar imágenes en subdirectorios
def find_image_path(filename):
    """Busca una imagen en los subdirectorios de images/"""
    base_image_dir = "images"

    # Buscar en todos los subdirectorios
    for root, dirs, files in os.walk(base_image_dir):
        if filename in files:
            return os.path.join(root, filename)

    return None


# Construir rutas completas buscando en los subdirectorios
paths = []
for filename in filenames:
    img_path = find_image_path(filename)
    if img_path:
        paths.append(img_path)
    else:
        print(f"[!] Advertencia: No se encontro el archivo {filename}")

print(f"[OK] Se encontraron {len(paths)} de {len(filenames)} archivos")


def load_gray_float(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return img_as_float(img)  # [0,1]


def compute_metrics(img_f):
    """
    Calcula métricas optimizadas para imágenes tipo retrato hablado (dibujo a lápiz).

    Ajustes para retratos hablados:
    - sigma más bajo para captar trazos finos
    - umbrales más bajos para detectar líneas suaves
    - mayor sensibilidad en detección de bordes
    """
    # Energy of Edges (Sobel) - Sin cambios, detecta gradientes en general
    grad = filters.sobel(img_f)  # [0,1]
    energy = float(np.sum(np.abs(grad)))

    # Densidad de trazos (Canny) - AJUSTADO MUY SENSIBLE para dibujos a lápiz
    # sigma=0.3: MUY sensible a trazos finos
    # low_threshold=0.02: Detecta hasta los trazos más suaves
    # high_threshold=0.10: Muy permisivo con bordes débiles
    edges = feature.canny(
        img_f,
        sigma=0.3,  # Minimo suavizado para captar todos los detalles
        low_threshold=0.02,  # Umbral muy bajo para captar trazos sutiles
        high_threshold=0.10,  # Umbral alto muy permisivo
    )
    stroke_density = float(edges.mean() * 100.0)

    # Contraste (0–255) - Mide diferencia entre trazos y fondo
    img255 = (img_f * 255.0).astype(np.float64)
    if edges.any():
        edge_mean = float(img255[edges].mean())
        non_edge_mean = float(img255[~edges].mean())
        contrast = edge_mean - non_edge_mean
    else:
        contrast = 0.0 - float(img255.mean())

    return energy, stroke_density, contrast


print("\n[*] Procesando imagenes y calculando metricas...")
records = []
for i, p in enumerate(paths, 1):
    print(f"   [{i}/{len(paths)}] Procesando: {os.path.basename(p)}")
    img = load_gray_float(p)
    e, d, c = compute_metrics(img)
    records.append(
        {"ID": os.path.basename(p), "Energy": e, "StrokeDensity_%": d, "Contrast": c}
    )

print("\n[*] Creando DataFrame y guardando CSV...")
df = pd.DataFrame(records)
df.to_csv("metrics_dodge_burn_50.csv", index=False)
print(f"[OK] CSV guardado: metrics_dodge_burn_50.csv")
print("\n[*] Primeras filas del DataFrame:")
print(df.head())  # vista rápida


# --- 3) Graficas (una por metrica) ---
print("\n[*] Generando graficas...")


def plot_metric(series, title, ylabel, outpng):
    xs = np.arange(1, len(series) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(xs, series.values, marker="o", linewidth=2, markersize=6)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Índice de imagen (1–50)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()
    print(f"   [OK] Grafica guardada: {outpng}")


plot_metric(
    df["Energy"], "Energía de bordes por imagen", "Energía (Σ|∇I|)", "plot_energy.png"
)
plot_metric(
    df["StrokeDensity_%"],
    "Densidad de trazos por imagen",
    "Densidad de trazos (%)",
    "plot_density.png",
)
plot_metric(
    df["Contrast"],
    "Contraste (bordes - fondo)",
    "Contraste (0–255)",
    "plot_contrast.png",
)

print("\n[OK] Proceso completado exitosamente!")

# --- 4) Interpretación sugerida (opcional) ---
print(
    "\nObjetivo de densidad: 15–30% es un rango óptimo para trazos definidos sin sobre-detección."
)
