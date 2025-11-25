"""
Script para visualizar los resultados de métricas.py
Muestra las gráficas y la tabla CSV generadas.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Verificar que los archivos existan
archivos_requeridos = [
    "metrics_dodge_burn_50.csv",
    "plot_energy.png",
    "plot_density.png",
    "plot_contrast.png",
]

print("🔍 Verificando archivos...")
for archivo in archivos_requeridos:
    if os.path.exists(archivo):
        print(f"   ✅ {archivo}")
    else:
        print(f"   ❌ {archivo} - NO ENCONTRADO")

# 1. Mostrar el CSV como tabla
print("\n" + "=" * 80)
print("📊 TABLA DE MÉTRICAS (metrics_dodge_burn_50.csv)")
print("=" * 80)

df = pd.read_csv("metrics_dodge_burn_50.csv")
print(f"\nTotal de imágenes analizadas: {len(df)}")
print(f"\nPrimeras 10 filas:")
print(df.head(10).to_string(index=False))

print(f"\n\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
print(df.describe())

# 2. Mostrar las gráficas en una ventana
print("\n" + "=" * 80)
print("📈 GRÁFICAS GENERADAS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Visualización de Métricas - 50 Imágenes", fontsize=16, fontweight="bold")

# Cargar y mostrar cada gráfica
graficas = [
    ("plot_energy.png", "Energía de Bordes"),
    ("plot_density.png", "Densidad de Trazos"),
    ("plot_contrast.png", "Contraste"),
]

for idx, (archivo, titulo) in enumerate(graficas):
    if os.path.exists(archivo):
        img = mpimg.imread(archivo)
        ax = axes[idx // 2, idx % 2]
        ax.imshow(img)
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.axis("off")
        print(f"   ✅ Mostrando: {archivo}")

# Mostrar tabla resumen en el cuarto panel
ax = axes[1, 1]
ax.axis("off")
resumen_text = f"""
RESUMEN DE MÉTRICAS
{'='*40}

📊 Energía de Bordes:
   Media: {df['Energy'].mean():.2f}
   Desv. Est: {df['Energy'].std():.2f}
   Rango: [{df['Energy'].min():.2f}, {df['Energy'].max():.2f}]

📊 Densidad de Trazos (%):
   Media: {df['StrokeDensity_%'].mean():.2f}%
   Desv. Est: {df['StrokeDensity_%'].std():.2f}%
   Rango: [{df['StrokeDensity_%'].min():.2f}%, {df['StrokeDensity_%'].max():.2f}%]

📊 Contraste:
   Media: {df['Contrast'].mean():.2f}
   Desv. Est: {df['Contrast'].std():.2f}
   Rango: [{df['Contrast'].min():.2f}, {df['Contrast'].max():.2f}]

✨ Objetivo densidad: 15-30% óptimo
"""
ax.text(
    0.1,
    0.5,
    resumen_text,
    fontsize=11,
    family="monospace",
    verticalalignment="center",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
print("\n🖼️  Mostrando ventana con todas las gráficas...")
print("   (Cierra la ventana para continuar)\n")
plt.show()

print("✨ Visualización completada!")
