# Parte 3 — Aplicación a función no trivial

---

## 1. Marco teórico

### Función de estudio

La función analizada es:

```
f(x) = x² + 2sin(3x)
```

Su derivada, necesaria para los métodos de optimización, es:

```
f'(x) = 2x + 6cos(3x)
```

Y la segunda derivada, requerida por Newton:

```
f''(x) = 2 − 18sin(3x)
```

### Propiedades relevantes

A diferencia de las funciones de la sección 2, `f` es **no convexa**: la componente `2sin(3x)` introduce oscilaciones que generan múltiples extremos locales. Esto la convierte en un caso de prueba significativo porque:

- Ningún método garantiza encontrar el mínimo global.
- El extremo al que converge cada método depende críticamente de las condiciones iniciales.
- `f''(x) = 2 − 18sin(3x)` oscila entre `−16` y `+20`, cambiando de signo frecuentemente.

### Comportamiento de f''

El signo de `f''(xn)` determina la dirección del paso de Newton: si `f''(xn) > 0` el paso va hacia la raíz de `f'` donde esta decrece (mínimo), y si `f''(xn) < 0` el paso va hacia donde `f'` crece (máximo). Dado que `f''` oscila de signo, Newton puede saltar entre regiones y no converger desde ciertos puntos iniciales.

---

## 2. Qué se hizo y cómo

### 2.1 Gráfica de f(x) y f'(x)

Se graficó `f(x)` y `f'(x)` en el intervalo `[−3, 5]` con 600 puntos para asegurar resolución suficiente. Los extremos de `f` se detectaron automáticamente buscando cambios de signo en `f'` con:

```python
cruces = np.where(np.diff(np.sign(df_vals)) != 0)[0]
```

El tipo de cada extremo se determina por el signo de `f'` justo antes del cruce:
- `df_vals[idx] < 0` → f' pasa de `−` a `+` → **mínimo local**
- `df_vals[idx] > 0` → f' pasa de `+` a `−` → **máximo local**

**Extremos detectados en `[−3, 5]`:**

| x ≈ | Tipo | f(x) ≈ |
|-----|------|--------|
| −2.332 | mínimo | 4.130 |
| −1.783 | máximo | 4.787 |
| −0.476 | mínimo | −1.753 |
|  0.589 | máximo | 2.309 |
|  1.408 | mínimo | 0.216 |

Se observa que en `[1.408, 5]` no hay más extremos: la componente lineal `2x` domina sobre `6cos(3x)` para `x` grande, dejando `f'(x) > 0` en toda esa región.

### 2.2 Predicciones previas

Antes de ejecutar los métodos, se predijo el resultado de cada combinación método/condición inicial usando el análisis de signos de `f'` y `f''` en los puntos iniciales. Las predicciones se documentaron en una celda Markdown del notebook.

**Criterio para cada método:**

- **Bisección:** se verifica si `f'(a) · f'(b) < 0` y se analiza qué cambio de signo de `f'` queda contenido en el intervalo.
- **Newton:** se evalúa el signo de `f''(x0)` para determinar la dirección del primer paso. Como `f''` puede ser negativo, Newton puede alejarse del extremo más cercano.
- **Descenso por gradiente:** se evalúa el signo de `f'(x0)`: si es positivo, el paso va a la izquierda; si es negativo, a la derecha. Se predice el extremo más cercano en esa dirección.

### 2.3 Experimentos

Se ejecutaron los tres métodos con las condiciones iniciales especificadas en la consigna. Para descenso por gradiente la tabla principal usó `lr ∈ {1e-3, 1e-2, 5e-2}`; el análisis comparativo extendido (sección 4.4) probó además `lr ∈ {1e-4, 1e-1, 3e-1}` desde `x0 = 0.585`.

---

## 3. Cómo se cumple la consigna

| Requisito de la consigna | Cumplimiento |
|---|---|
| Graficar f(x) en un intervalo adecuado | ✓ Graficada en `[−3, 5]` con f'(x), y=0 y marcas verticales en cada extremo |
| Analizar condiciones iniciales y predecir resultado | ✓ Celda Markdown con predicción analítica para cada combinación |
| Ejecutar métodos para cada condición inicial | ✓ Bisección ×3, Newton ×3, descenso ×9 (3 x0 × 3 lr) |
| Generar tabla con método, predicción, resultado e iteraciones | ✓ DataFrame con columnas Método / Condición / Predicción / xn / f(xn) / Iters |
| Probar varios learning rates en descenso y analizar | ✓ `lr ∈ {1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1}` con análisis de lr pequeño, grande y rango seguro |
| Responder preguntas de la consigna (sección 3.3) | ✓ Celda Markdown con análisis caso por caso, maximización con −f y análisis de riesgo |
| **BONUS:** devolver sucesión de aproximaciones y graficar con código de colores | ✓ Copias `biseccion_hist`, `newton_hist`, `descenso_hist`; grilla 3×2 con colormap `plasma` |

---

## 4. Resultados

### 4.1 Bisección

| Intervalo | Predicción | xn | f(xn) | Iters | ¿Correcta? |
|-----------|------------|----|-------|-------|------------|
| `[−2, 1]` | máximo x≈−1.783 | −1.7829 | 4.7873 | 22 | ✓ |
| `[−1, 0]` | mínimo x≈−0.476 | −0.4710 | −1.7533 | 20 | ✓ |
| `[1, 2]`  | mínimo x≈1.408  |  1.4080 |  0.2163 | 20 | ✓ |

Las tres predicciones fueron correctas. En cada caso el intervalo contenía un único cambio de signo de `f'`, lo que elimina toda ambigüedad sobre el extremo a encontrar.

### 4.2 Newton

| x0 | Predicción | xn | f(xn) | Iters | ¿Correcta? |
|----|------------|----|-------|-------|------------|
| −2 | máximo x≈−1.783 | −1.7829 | 4.7873 | 5 | ✓ |
| −1 | máximo x≈0.589  |  0.5895 | 2.3086 | 5 | ✓ |
|  4 | diverge         |  7.8209 | 59.176 | 1000 | ✓ |

**Casos notables:**

- `x0 = −2`: aunque el mínimo más cercano es x≈−2.332, el primer paso va a la derecha porque `f''(−2) ≈ −3.03 < 0`. Newton da x₁ = −2 − (+1.76)/(−3.03) ≈ −1.42 y converge al máximo en −1.783 en solo 5 iteraciones. Newton no garantiza converger al extremo geográficamente más próximo, sino al que resulta de la dinámica f'/f''.

- `x0 = 4`: Newton oscila en la región `[2.8, 3.5]` donde `f''` cambia de signo frecuentemente. Los pasos no se amortiguan y el método diverge progresivamente hacia x≈7.82 tras agotar las 1000 iteraciones.

### 4.3 Descenso por gradiente

| x0 | lr | Predicción | xn | f(xn) | Iters | ¿Correcta? |
|----|----|------------|----|-------|-------|------------|
| −3 | 1e-3 | mínimo x≈−2.332 | −2.3229 | 4.1297 | 636 | ✓ |
| −3 | 1e-2 | mínimo x≈−2.332 | −2.3228 | 4.1297 |  77 | ✓ |
| −3 | 5e-2 | mínimo x≈−2.332 | −2.3228 | 4.1297 |  12 | ✓ |
| 0.585 | 1e-3 | mínimo x≈−0.476 | −0.4710 | −1.7533 | 795 | ✓ |
| 0.585 | 1e-2 | mínimo x≈−0.476 | −0.4710 | −1.7533 |  89 | ✓ |
| 0.585 | 5e-2 | mínimo x≈−0.476 | −0.4710 | −1.7533 |  14 | ✓ |
| 4 | 1e-3 | mínimo x≈1.408 | 2.9308 | 9.7717 | 1000 | ✗ |
| 4 | 1e-2 | mínimo x≈1.408 | 1.4080 | 0.2163 | 180 | ✓ |
| 4 | 5e-2 | mínimo x≈1.408 | 1.4080 | 0.2163 |  30 | ✓ |

**Casos notables:**

- `x0 = 4, lr = 1e-3`: el gradiente en x=4 es f'(4)≈13, pero con lr tan pequeño cada paso es de ≈0.013. La distancia hasta el mínimo en x≈1.408 es de ≈2.6 unidades y las 1000 iteraciones no alcanzan; el método queda detenido en x≈2.93.

- `x0 = 0.585`: el gradiente inicial es casi nulo (f'(0.585)≈0.07), lo que implica pasos pequeños. Sin embargo, todos los lr convergen al mismo mínimo; el efecto del lr es en **velocidad**, no en el extremo alcanzado.

### 4.4 Análisis de learning rates

Experimento con `x0 = 0.585` y seis learning rates:

| lr | xn | f(xn) | Iters | ¿Convergió? |
|----|-----|-------|-------|-------------|
| `1e-4` | 0.5678 | 2.3048 | 1000 | ✗ |
| `1e-3` | −0.4710 | −1.7533 | 795 | ✓ |
| `1e-2` | −0.4710 | −1.7533 | 89 | ✓ |
| `5e-2` | −0.4710 | −1.7533 | 14 | ✓ |
| `1e-1` | −0.4710 | −1.7533 | 511 | ✓ |
| `3e-1` | 0.8932 | 1.6892 | 1000 | ✗ |

**lr muy pequeño (`1e-4`):** el gradiente en `x0 = 0.585` es casi nulo (`f'(0.585) ≈ 0.07`), por lo que cada paso vale `1e-4 × 0.07 ≈ 7×10⁻⁶`, apenas por encima de la tolerancia `1e-6`. El criterio de parada nunca se activa y en 1000 iteraciones el método solo recorre `≈0.017` unidades desde el punto inicial.

**lr muy grande (`3e-1`):** el primer paso es pequeño (gradiente casi nulo), pero al atravesar la zona del máximo en `x ≈ 0.589`, el gradiente cambia de signo bruscamente y los pasos de `0.3 × |gradiente|` son suficientemente grandes para saltar el mínimo en `−0.476`. El método queda oscilando en otra región y termina en `x ≈ 0.893` sin convergir.

**Rango seguro:** para este punto inicial, `lr ∈ [1e-3, 1e-1]` converge al mínimo correcto, pero con velocidades muy distintas. El óptimo empírico es `lr = 5e-2` (14 iters); con `lr = 1e-1` el método oscila alrededor del mínimo antes de estabilizarse (511 iters). No existe un rango universalmente seguro: el mismo `lr = 1e-1` desde `x0 = 4` (donde `f'(4) ≈ 13`) produce pasos de `1.3`, potencialmente saltando el mínimo.

---

## 5. Respuestas a las preguntas de la consigna (sección 3.3)

### ¿Los métodos funcionaron como se esperaba para las condiciones iniciales?

**Bisección — todos correctos.** El intervalo determinó sin ambigüedad el extremo en cada caso.

**Newton — dos correctos, uno divergió como se predijo:**

- `x0 = −2` → máximo x≈−1.783 ✓. Se predijo correctamente: `f''(−2) < 0` hace que el primer paso vaya a la derecha, no hacia el mínimo en −2.332.
- `x0 = −1` → máximo x≈0.589 ✓. El paso inicial es grande y cae en la cuenca del máximo en 0.589.
- `x0 = 4` → diverge ✓. `f''` cambia de signo repetidamente en esa región; Newton no encuentra dirección estable.

**Descenso — ocho correctos, uno no convergió:**

- Todos los casos con `x0 ∈ {−3, 0.585}` y todos los lr convergieron al mínimo correcto ✓.
- `x0 = 4, lr = 1e-3` → no convergió ✗. No se predijo explícitamente. Con pasos de `≈0.013` y una distancia de `≈2.6` unidades hasta el mínimo, las 1000 iteraciones son insuficientes. Ilustra que un lr demasiado pequeño puede ser igual de problemático que uno demasiado grande.
- `x0 = 0.585, lr = 1e-4` → no convergió en el análisis extendido ✗. El paso es del orden de la tolerancia; el criterio de parada nunca se activa.

### ¿Cómo usar descenso por gradiente para obtener máximos sin cambiar el algoritmo?

Maximizar `f(x)` es equivalente a minimizar `−f(x)`:

```
argmax f(x) = argmin −f(x)
```

Como `(−f)'(x) = −f'(x)`, el descenso sobre `−f` actualiza en la dirección opuesta al gradiente de `f`, ascendiendo por ella:

```
x_{n+1} = x_n − lr · (−f'(x_n)) = x_n + lr · f'(x_n)
```

Con el sistema simbólico del proyecto basta pasar la función negada como argumento:

```python
descenso_gradiente(-f, x0, lr)
```

El algoritmo no se modifica en absoluto.

### ¿Hay riesgo de no convergencia buscando máximos para x0 ∈ {−3, 0.585, 4}?

Aplicando descenso sobre `−f`, el método sigue `+f'(x)`, es decir, asciende por `f`:

- **x0 = −3**: `f'(−3) ≈ −11.47` → el ascenso mueve hacia la derecha. El primer máximo en esa dirección es x≈−1.783. **Converge.**

- **x0 = 0.585**: `f'(0.585) ≈ +0.07` → se mueve ligeramente a la derecha hacia el máximo en x≈0.589. **Converge**, aunque lento por gradiente casi nulo al inicio.

- **x0 = 4**: `f'(4) ≈ +13` → el ascenso mueve hacia la derecha. Para `x > 1.5` no existen más máximos: `f'(x) > 0` para `x` grande implica que el método seguirá moviéndose a la derecha indefinidamente. **Hay riesgo real de no convergencia** (divergencia hacia `+∞`).

---

## 6. BONUS — Evolución de aproximaciones (sección 3.2)

Se implementaron copias de los tres métodos (`biseccion_hist`, `newton_hist`, `descenso_hist`) que devuelven, además del resultado, la lista completa de aproximaciones en cada iteración. Los algoritmos originales no fueron modificados.

### Implementación

Cada copia agrega una lista `history` que registra los valores `xn` generados en cada paso:

- **Bisección:** registra cada punto medio `m` calculado.
- **Newton y descenso:** incluyen `x0` en el historial y agregan cada `x_new`.

### Visualización

Se generó una grilla 3×2 con seis casos de estudio sobre `f(x) = x² + 2sin(3x)`:

| Fila | Caso izquierdo | Caso derecho |
|------|---------------|--------------|
| Bisección | `[−2, 1]` → máximo x≈−1.783 (22 iters) | `[−1, 0]` → mínimo x≈−0.476 (20 iters) |
| Newton | `x0=−2` → máximo x≈−1.783 (5 iters) | `x0=4` → diverge hacia x≈7.82 (1000 iters) |
| Descenso | `x0=0.585, lr=5e-2` → mínimo x≈−0.476 (14 iters) | `x0=0.585, lr=1e-1` → mínimo x≈−0.476 (511 iters) |

Cada subplot muestra la curva `f(x)` como fondo y los puntos de la sucesión de aproximaciones coloreados con el colormap `plasma` (oscuro = primera iteración, amarillo = última). Para historiales de más de 50 puntos se submuestrean 50 igualmente espaciados para mantener la legibilidad.

**Aspectos visuales de interés:**

- **Bisección:** los puntos convergen desde dos lados alternados hacia el extremo, reflejando el mecanismo de reducción de intervalo.
- **Newton x0=−2:** solo 5 puntos visibles (incluido x0), todos muy juntos — ilustra la convergencia cuadrática.
- **Newton x0=4:** los 50 puntos submuestreados de las 1000 iteraciones se dispersan ampliamente a la derecha de la curva principal, mostrando la divergencia progresiva.
- **Descenso lr=5e-2 vs lr=1e-1:** el primero converge limpiamente en 14 pasos; el segundo oscila visiblemente antes de estabilizarse (511 pasos), ambos al mismo mínimo.
