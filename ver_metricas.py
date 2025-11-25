import pandas as pd

df = pd.read_csv("metrics_dodge_burn_50.csv")

print("=" * 80)
print("📊 RESULTADOS CON PARÁMETROS AJUSTADOS PARA RETRATOS HABLADOS")
print("=" * 80)
print(f"\nTotal de imágenes: {len(df)}\n")

print("📈 ESTADÍSTICAS:")
print("-" * 80)
print(f"Energía de Bordes (Sobel):")
print(f"  Media: {df['Energy'].mean():.2f}")
print(f"  Desv. Std: {df['Energy'].std():.2f}")
print(f"  Rango: [{df['Energy'].min():.2f}, {df['Energy'].max():.2f}]\n")

print(f"Densidad de Trazos (%):")
print(f"  Media: {df['StrokeDensity_%'].mean():.2f}%")
print(f"  Desv. Std: {df['StrokeDensity_%'].std():.2f}%")
print(
    f"  Rango: [{df['StrokeDensity_%'].min():.2f}%, {df['StrokeDensity_%'].max():.2f}%]"
)
print(f"  ✅ Objetivo: 15-30% para trazos bien definidos\n")

print(f"Contraste (trazos - fondo):")
print(f"  Media: {df['Contrast'].mean():.2f}")
print(f"  Desv. Std: {df['Contrast'].std():.2f}")
print(f"  Rango: [{df['Contrast'].min():.2f}, {df['Contrast'].max():.2f}]\n")

print("=" * 80)
print("📋 PRIMERAS 10 IMÁGENES:")
print("=" * 80)
print(df.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("📋 ÚLTIMAS 10 IMÁGENES:")
print("=" * 80)
print(df.tail(10).to_string(index=False))
