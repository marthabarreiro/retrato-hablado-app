# Flujo de Procesamiento de Imágenes para Retrato Hablado
## Documento Técnico-Científico

---

## 1. Introducción

Este documento describe el flujo matemático completo del procesamiento de imágenes faciales en el sistema de retrato hablado, desde que las imágenes están cargadas en memoria hasta la generación del boceto final al seleccionar cada parte facial con el botón "Aplicar".

---

## 2. Arquitectura del Sistema

### 2.1 Estructura de Datos en Memoria

Las imágenes faciales se almacenan en un catálogo estructurado como DataFrame de pandas con los siguientes campos:

```
{
    "tipo": str,           // "tipo_rostro" o "parte_facial"
    "genero": str,         // "hombre", "mujer", "unisex"
    "parte": str,          // "tipo_rostro", "ojos", "nariz", "boca", etc.
    "codigo": str,         // Identificador único (e.g., "tp001", "oj023")
    "descripcion": str,    // Descripción textual de la característica
    "imagen": str,         // Ruta al archivo de imagen
    "indices": dict        // Puntos de referencia geométricos {nombre: {x, y}}
}
```

### 2.2 Flujo de Ejecución al Presionar "Aplicar"

```
Usuario presiona "Aplicar"
    ↓
custom_widgets.py: on_button_click()
    ↓
main.py: on_face_image_added(code)
    ↓
main.py: create_face_sketch()
    ↓
tools.py: suma_imagenes_dominio_frecuencial(face_parts_data)
    ↓
Retorno y visualización del boceto
```

---

## 3. Pipeline de Procesamiento Matemático

### 3.1 Fase 1: Preprocesamiento de Imágenes

#### 3.1.1 Carga de la Plantilla Base

La plantilla base (tipo de rostro) se carga en escala de grises:

```
I_plantilla: ℝ^(h×w) → [0, 255]
```

donde:
- `h` = altura en píxeles
- `w` = ancho en píxeles

Esta plantilla define la estructura geométrica base del rostro mediante un conjunto de puntos de referencia:

```
P_plantilla = {p_i : (x_i, y_i) | i ∈ {1, ..., n}}
```

#### 3.1.2 Carga de Partes Faciales

Cada parte facial `k` se carga en escala de grises:

```
I_k: ℝ^(h_k×w_k) → [0, 255]
```

con su propio conjunto de puntos de referencia:

```
P_k = {p_i : (x_i^k, y_i^k) | i ∈ {1, ..., n}}
```

**Nota especial para orejas:** Las orejas se procesan de forma dual, generando dos imágenes:
- Original: `I_oreja`
- Reflejada: `I_oreja_flip = flip_horizontal(I_oreja)`

---

### 3.2 Fase 2: Homologación de Tono de Piel (Opcional, si hay ≥4 partes)

Esta fase aplica el **método de transferencia de color de Reinhard** para homogeneizar el tono de piel entre diferentes partes faciales.

#### 3.2.1 Detección de Piel mediante Segmentación Multi-espacio

Se utiliza una combinación de dos espacios de color para máxima robustez:

**a) Espacio HSV (Hue, Saturation, Value):**

```
I_HSV = RGB2HSV(I_BGR)
H, S, V = split(I_HSV)

M_HSV₁ = {1 si (0 ≤ H ≤ 25) ∧ (40 ≤ S ≤ 255) ∧ (0 ≤ V ≤ 255)
          0 en otro caso}

M_HSV₂ = {1 si (165 ≤ H ≤ 180) ∧ (40 ≤ S ≤ 255) ∧ (0 ≤ V ≤ 255)
          0 en otro caso}

M_HSV = M_HSV₁ ∨ M_HSV₂
```

**b) Espacio YCrCb (Luminancia-Crominancia):**

```
I_YCrCb = BGR2YCrCb(I_BGR)
Y, Cr, Cb = split(I_YCrCb)

M_YCrCb = {1 si (0 ≤ Y ≤ 255) ∧ (135 ≤ Cr ≤ 180) ∧ (85 ≤ Cb ≤ 135)
           0 en otro caso}
```

**c) Máscara combinada:**

```
M_piel = M_HSV ∧ M_YCrCb
```

**d) Refinamiento mediante filtros morfológicos:**

```
M_piel_refinada = GaussianBlur(MedianBlur(M_piel, k=7), σ=9)
```

**e) Eliminación de regiones oscuras (ojos, labios):**

```
I_gray = BGR2Gray(I_BGR)
M_oscuras = {1 si I_gray > 200
             0 en otro caso}

M_oscuras_suavizada = GaussianBlur(M_oscuras, σ=17)

M_piel_final = M_piel_refinada ∧ ¬M_oscuras_suavizada
```

#### 3.2.2 Transferencia de Color de Reinhard

Este método ajusta las estadísticas de color de la imagen fuente para que coincidan con la imagen de referencia en el espacio LAB.

**Transformación al espacio LAB:**

```
I_LAB = BGR2LAB(I_BGR)
L, A, B = split(I_LAB)
```

**Extracción de píxeles de piel:**

```
A_src = {A(x,y) | M_piel_src(x,y) = 1}
B_src = {B(x,y) | M_piel_src(x,y) = 1}

A_tgt = {A(x,y) | M_piel_tgt(x,y) = 1}
B_tgt = {B(x,y) | M_piel_tgt(x,y) = 1}
```

**Cálculo de estadísticas:**

```
μ_A_src = E[A_src],  σ_A_src = √Var[A_src]
μ_B_src = E[B_src],  σ_B_src = √Var[B_src]

μ_A_tgt = E[A_tgt],  σ_A_tgt = √Var[A_tgt]
μ_B_tgt = E[B_tgt],  σ_B_tgt = √Var[B_tgt]
```

**Ajuste de canales de crominancia:**

Para cada píxel en la región de piel:

```
A'(x,y) = ((A(x,y) - μ_A_src) × (σ_A_tgt / σ_A_src)) + μ_A_tgt

B'(x,y) = ((B(x,y) - μ_B_src) × (σ_B_tgt / σ_B_src)) + μ_B_tgt

L'(x,y) = L(x,y)  // La luminancia no se modifica
```

**Reconstrucción de la imagen:**

```
I'_LAB = merge(L', A', B')
I'_BGR = LAB2BGR(I'_LAB)
```

**Suavizado final mediante filtro bilateral:**

```
I_final = BilateralFilter(I'_BGR, d=7, σ_color=50, σ_space=50)
```

El filtro bilateral preserva bordes mientras suaviza:

```
I_final(x) = (1/W(x)) ∑_{y∈Ω} I'_BGR(y) × exp(-‖x-y‖²/2σ²_space) × exp(-‖I(x)-I(y)‖²/2σ²_color)

donde W(x) = ∑_{y∈Ω} exp(-‖x-y‖²/2σ²_space) × exp(-‖I(x)-I(y)‖²/2σ²_color)
```

---

### 3.3 Fase 3: Generación de Boceto mediante Dodge & Burn

Esta técnica simula el proceso fotográfico tradicional de aclarado y oscurecimiento para crear un efecto de dibujo a lápiz.

#### 3.3.1 Algoritmo Dodge & Burn

Para cada imagen de parte facial en escala de grises `I`:

**Paso 1: Inversión de la imagen (negativo)**

```
I_inv(x,y) = 255 - I(x,y)
```

**Paso 2: Desenfoque gaussiano del negativo**

```
I_blur(x,y) = ∑_{i=-k}^{k} ∑_{j=-k}^{k} I_inv(x+i, y+j) × G(i,j)

donde G(i,j) = (1/(2πσ²)) × exp(-(i²+j²)/(2σ²))
```

Con kernel de tamaño `21×21` y `σ` calculado automáticamente por OpenCV.

**Paso 3: División de color (dodge blending)**

```
I_boceto(x,y) = 256 × I(x,y) / (255 - I_blur(x,y) + ε)
```

donde `ε = 10⁻⁶` es una constante pequeña para evitar división por cero.

**Paso 4: Normalización**

```
I_final(x,y) = clip(I_boceto(x,y), 0, 255)
```

**Interpretación matemática:**

La división amplifica las regiones donde el negativo desenfocado es bajo (bordes y detalles), creando el efecto de lápiz. Las regiones uniformes se atenúan debido al denominador alto.

---

### 3.4 Fase 4: Aplicación de Máscara Suavizada

Para evitar bordes duros al combinar partes faciales, se aplica una máscara con transición suave hacia blanco en los bordes.

#### 3.4.1 Creación de Máscara Gaussiana

**Paso 1: Definir región de bordes**

```
M_bordes(x,y) = {255 si (x < p) ∨ (x ≥ w-p) ∨ (y < p) ∨ (y ≥ h-p)
                 0   en otro caso}
```

donde `p` = padding (típicamente 5 píxeles)

**Paso 2: Suavizado gaussiano**

```
M_suavizada = GaussianBlur(M_bordes, k=(2p+1)×(2p+1), σ=0)

k_size = 2p + 1 = 11 (para p=5)
```

**Paso 3: Suma ponderada con la imagen**

```
I_masked(x,y) = clip(I(x,y) + M_suavizada(x,y), 0, 255)
```

**Efecto:** Los bordes de la imagen se funden gradualmente hacia blanco (255), facilitando la composición sin artefactos visuales.

---

### 3.5 Fase 5: Transformación Geométrica y Colocación

Esta fase alinea cada parte facial con la plantilla base mediante transformación afín basada en puntos de referencia.

#### 3.5.1 Cálculo del Factor de Escala

Dados conjuntos de puntos coincidentes:
- Plantilla: `{p_i^T = (x_i^T, y_i^T) | i ∈ {1, ..., n}}`
- Parte: `{p_i^P = (x_i^P, y_i^P) | i ∈ {1, ..., n}}`

**Distancias euclidianas entre pares de puntos:**

```
d_ij^T = ‖p_i^T - p_j^T‖ = √((x_i^T - x_j^T)² + (y_i^T - y_j^T)²)

d_ij^P = ‖p_i^P - p_j^P‖ = √((x_i^P - x_j^P)² + (y_i^P - y_j^P)²)
```

**Factor de escala promedio:**

```
s = (1/C(n,2)) × ∑_{i=1}^{n-1} ∑_{j=i+1}^{n} (d_ij^T / d_ij^P)

donde C(n,2) = n(n-1)/2 es el número de pares
```

#### 3.5.2 Redimensionamiento de la Parte Facial

Aplicación de interpolación bilineal:

```
I_escalada(x,y) = InterpolaciónBilineal(I_parte, s×x, s×y)
```

Para interpolación bilineal:

```
I(x',y') ≈ (1-α)(1-β)I(⌊x'⌋,⌊y'⌋) + α(1-β)I(⌈x'⌉,⌊y'⌋) 
         + (1-α)βI(⌊x'⌋,⌈y'⌉) + αβI(⌈x'⌉,⌈y'⌉)

donde α = frac(x'), β = frac(y')
```

#### 3.5.3 Cálculo del Vector de Traslación

Para cada punto de referencia `i`:

```
t_i = p_i^T - s × p_i^P
```

**Traslación promedio:**

```
t̄ = (1/n) × ∑_{i=1}^{n} t_i = (t̄_x, t̄_y)
```

#### 3.5.4 Composición Final

Para cada píxel `(x,y)` de la parte escalada:

```
(x_T, y_T) = (x + t̄_x, y + t̄_y)

Si (x_T, y_T) ∈ [0, w_T) × [0, h_T) y I_escalada(x,y) < 250:
    I_resultado(x_T, y_T) = I_escalada(x,y)
```

**Criterio de 250:** Solo se copian píxeles no blancos, preservando transparencia.

---

### 3.6 Fase 6: Suma en el Dominio Frecuencial

Esta es la fase clave que combina todas las partes faciales mediante análisis de Fourier.

#### 3.6.1 Transformada de Fourier 2D (FFT)

Para cada parte facial colocada `I_k` (ya con boceto, máscara y alineada):

**Transformada de Fourier discreta 2D:**

```
F_k(u,v) = ∑_{x=0}^{w-1} ∑_{y=0}^{h-1} I_k(x,y) × exp(-2πi(ux/w + vy/h))
```

Donde:
- `(u,v)` son las coordenadas en el dominio frecuencial
- `i` es la unidad imaginaria
- `F_k(u,v)` es un número complejo: `F_k = R_k + iI_k`

**Desplazamiento al centro (fftshift):**

```
F_k_centrada(u,v) = F_k((u + w/2) mod w, (v + h/2) mod h)
```

Este desplazamiento coloca las bajas frecuencias (componentes suaves) en el centro del espectro.

#### 3.6.2 Suma en el Dominio Frecuencial

```
F_suma(u,v) = ∑_{k=1}^{N} F_k_centrada(u,v)
```

**Interpretación física:**

En el dominio frecuencial, la suma es equivalente a la convolución en el dominio espacial. Las frecuencias comunes se refuerzan, mientras que las diferencias se atenúan. Esto produce una composición más natural que la simple superposición de píxeles.

**Propiedades importantes:**

1. **Linealidad:** `ℱ{a×f + b×g} = a×ℱ{f} + b×ℱ{g}`
2. **Preservación de energía (Parseval):** `∑|I(x,y)|² = (1/wh)∑|F(u,v)|²`
3. **Simetría hermítica:** Para imágenes reales, `F(-u,-v) = F*(u,v)`

#### 3.6.3 Transformada Inversa de Fourier (IFFT)

**Desplazamiento inverso:**

```
F_suma_desplazada(u,v) = ifftshift(F_suma(u,v))
```

**Transformada inversa:**

```
I_resultado(x,y) = (1/(w×h)) × ∑_{u=0}^{w-1} ∑_{v=0}^{h-1} F_suma(u,v) × exp(2πi(ux/w + vy/h))
```

**Magnitud (parte real):**

```
I_boceto(x,y) = |I_resultado(x,y)| = √(Re[I_resultado]² + Im[I_resultado]²)
```

---

### 3.7 Fase 7: Normalización Final

#### 3.7.1 Normalización al Rango [0, 255]

La imagen resultante de la IFFT puede tener valores fuera del rango visible:

```
I_min = min{I_boceto(x,y) | ∀x,y}
I_max = max{I_boceto(x,y) | ∀x,y}
```

**Normalización lineal:**

```
I_normalizada(x,y) = 255 × (I_boceto(x,y) - I_min) / (I_max - I_min)
```

**Conversión a enteros de 8 bits:**

```
I_final(x,y) = ⌊I_normalizada(x,y) + 0.5⌋ ∈ [0, 255] ⊂ ℤ
```

---

## 4. Análisis Matemático del Dominio Frecuencial

### 4.1 ¿Por qué sumar en frecuencias?

**Ventaja 1: Reducción de Artefactos de Borde**

En el dominio espacial, la superposición directa de imágenes crea discontinuidades:

```
I_directa = I_1 + I_2  →  discontinuidades visibles en transiciones
```

En el dominio frecuencial, los componentes de alta frecuencia (bordes) se modulan mejor:

```
ℱ{I_1 + I_2} = ℱ{I_1} + ℱ{I_2}  →  mezcla más suave de características
```

**Ventaja 2: Descomposición en Componentes de Frecuencia**

La FFT descompone la imagen en componentes senoidales:

```
I(x,y) = ∑_{u,v} A(u,v) × cos(2π(ux/w + vy/h)) + B(u,v) × sin(2π(ux/w + vy/h))
```

Donde:
- **Bajas frecuencias** (centro del espectro): Formas globales, tono general
- **Altas frecuencias** (periferia del espectro): Detalles finos, bordes

La suma en frecuencias permite que las características de diferentes escalas se combinen de forma coherente.

### 4.2 Complejidad Computacional

**FFT 2D:**
```
T_FFT = O(w × h × log(w × h))
```

**Suma:**
```
T_suma = O(w × h)
```

**IFFT 2D:**
```
T_IFFT = O(w × h × log(w × h))
```

**Complejidad total:**
```
T_total = N × T_FFT + T_suma + T_IFFT = O(N × w × h × log(w × h))
```

donde `N` es el número de partes faciales (típicamente 5-10).

---

## 5. Flujo Completo Paso a Paso: Ejemplo Concreto

### 5.1 Estado Inicial

Usuario ha seleccionado:
1. Tipo de rostro: `tp001` (plantilla base)
2. Ojos: `oj023`
3. Nariz: `nz015`
4. Boca: `bc008`
5. Cejas: `cj012`

### 5.2 Ejecución al Presionar "Aplicar" en Boca

**T0:** Clic en botón "Aplicar" de `bc008`
```python
on_button_click(instance) → on_face_image_added("bc008")
```

**T1:** Actualización del array de partes
```python
face_images.append({"part": "boca", "code": "bc008"})
```

**T2:** Inicio de procesamiento
```python
create_face_sketch() → suma_imagenes_dominio_frecuencial(face_parts_data)
```

**T3:** Carga de datos
```python
parts_data = [
    {parte: "tipo_rostro", imagen: "images/plantillas/tp001.png", indices: {...}},
    {parte: "ojos", imagen: "images/ojos/oj023.png", indices: {...}},
    {parte: "nariz", imagen: "images/nariz/nz015.png", indices: {...}},
    {parte: "boca", imagen: "images/boca/bc008.png", indices: {...}},
    {parte: "cejas", imagen: "images/cejas/cj012.png", indices: {...}}
]
```

**T4:** Homologación de tono de piel (si ≥4 partes)
```python
# Cargar imágenes en BGR
imgs_bgr = [cv.imread(path) for path in paths[1:]]  # Excluye plantilla

# Aplicar método de Reinhard
imgs_same_skin = match_skin_tone(imgs_bgr, ref_index=3)  # boca como referencia

# Resultado: todas las partes tienen el mismo tono de piel
```

**T5:** Procesamiento de cada parte
```python
for parte in partes_faciales:
    # 5.1 Convertir a escala de grises
    img_gray = cv.imread(parte["imagen"], cv.IMREAD_GRAYSCALE)
    
    # 5.2 Aplicar filtro dodge para boceto
    img_boceto = filtro_dodge(img_gray)
    
    # 5.3 Aplicar máscara suavizada
    img_masked = aplica_mascara_cuadrada_suavizada(img_boceto, padding=5)
    
    # 5.4 Colocar en plantilla con transformación geométrica
    img_colocada = colocar_parte_en_plantilla(
        plantilla["imagen"],
        img_masked,
        plantilla["indices"][parte["parte"]],
        parte["indices"]
    )
    
    # 5.5 FFT y acumulación
    F_parte = fftshift(fft2(img_colocada))
    F_suma += F_parte
```

**T6:** Síntesis final
```python
# 6.1 Transformada inversa
img_resultado = abs(ifft2(ifftshift(F_suma)))

# 6.2 Normalización
img_normalizada = normalizar_grises(img_resultado)

# 6.3 Flip vertical para Kivy
img_final = cv.flip(img_normalizada, 0)
```

**T7:** Visualización
```python
# Crear textura de Kivy
texture = Texture.create(size=(width, height), colorfmt="luminance")
texture.blit_buffer(img_final.tobytes(), colorfmt="luminance", bufferfmt="ubyte")

# Actualizar widget
self.root.ids.build_image.texture = texture
```

---

## 6. Métricas y Consideraciones Científicas

### 6.1 Métricas de Calidad

**PSNR (Peak Signal-to-Noise Ratio):**
```
MSE = (1/(w×h)) × ∑∑(I_referencia(x,y) - I_resultado(x,y))²

PSNR = 10 × log₁₀(255² / MSE) [dB]
```

**SSIM (Structural Similarity Index):**
```
SSIM(x,y) = ((2μ_xμ_y + C₁)(2σ_xy + C₂)) / ((μ_x² + μ_y² + C₁)(σ_x² + σ_y² + C₂))

donde:
μ_x, μ_y: medias locales
σ_x, σ_y: varianzas locales
σ_xy: covarianza
C₁, C₂: constantes de estabilización
```

### 6.2 Limitaciones y Consideraciones

1. **Pérdida de información de fase:** La normalización basada solo en magnitud descarta información de fase que podría contener detalles estructurales.

2. **Asunción de linealidad:** La suma frecuencial asume que las contribuciones son linealmente separables, lo cual es aproximado en imágenes naturales.

3. **Aliasing:** El redimensionamiento puede introducir aliasing si el factor de escala es significativamente diferente de 1.

4. **Efectos de borde:** A pesar de la máscara suavizada, pueden aparecer artefactos de Gibbs en transiciones abruptas.

---

## 7. Conclusiones

El sistema implementa un pipeline sofisticado que combina:

1. **Segmentación multi-espacio** para detección robusta de piel
2. **Transferencia de color de Reinhard** para homogeneización cromática
3. **Dodge & Burn** para efectos artísticos de boceto
4. **Transformaciones afines** para alineación geométrica precisa
5. **Suma en dominio frecuencial** para composición suave y natural
6. **Normalización adaptativa** para optimización del contraste

Este enfoque matemáticamente fundamentado produce retratos hablados de alta calidad con transiciones naturales y sin artefactos visuales significativos.

---

## Referencias Matemáticas

1. **Transformada de Fourier:** Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation*, 19(90), 297-301.

2. **Transferencia de color de Reinhard:** Reinhard, E., Adhikhmin, M., Gooch, B., & Shirley, P. (2001). Color transfer between images. *IEEE Computer Graphics and Applications*, 21(5), 34-41.

3. **Filtro bilateral:** Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and color images. *ICCV*, 839-846.

4. **Detección de piel:** Kakumanu, P., Makrogiannis, S., & Bourbakis, N. (2007). A survey of skin-color modeling and detection methods. *Pattern Recognition*, 40(3), 1106-1122.

5. **Dodge & Burn:** Alasdair McAndrew (2004). An Introduction to Digital Image Processing with Matlab. *Course Notes*.

---

**Fecha de elaboración:** Agosto 2026  
**Versión:** 1.0  
**Sistema:** Retrato Hablado Digital - Análisis de Procesamiento de Imágenes
