# Parte 4 — Ajuste de datos mediante optimización

## 1. Marco teórico

El problema de ajuste de modelos consiste en encontrar el parámetro $w$ que mejor describe un conjunto de datos $(x_i, y_i)$ según un modelo paramétrico $\hat{y}(x; w)$. La calidad del ajuste se mide con la función de pérdida de error cuadrático medio (MSE):

$$L(w) = \frac{1}{N} \sum_{i=1}^{N} \left(\hat{y}(x_i;\, w) - y_i\right)^2$$

El objetivo es encontrar $w^* = \arg\min_w L(w)$ usando descenso por gradiente:

$$w_{n+1} = w_n - \text{lr} \cdot L'(w_n)$$

donde $L'(w) = \frac{2}{N}\sum_{i=1}^N \left(\hat{y}(x_i; w) - y_i\right) \cdot \frac{\partial \hat{y}}{\partial w}(x_i; w)$.

Los tres modelos considerados y sus derivadas respecto de $w$ son:

| Modelo | $\hat{y}(x; w)$ | $\frac{\partial \hat{y}}{\partial w}$ |
|--------|-----------------|----------------------------------------|
| Lineal | $wx$ | $x$ |
| Exponencial | $e^{wx}$ | $x \cdot e^{wx}$ |
| Sinusoidal | $\sin(wx)$ | $x \cdot \cos(wx)$ |

---

## 2. Qué se implementó y cómo

### Construcción simbólica de `L(w)`

La función de pérdida se construyó utilizando **exclusivamente el sistema de diferenciación simbólica** del proyecto (`abstractions/`). La función `build_loss` itera sobre los datos del CSV, construye cada término $(\hat{y}(x_i; w) - y_i)^2$ como un árbol de objetos `Function`, y los acumula sumándolos con el operador `+`:

```python
def build_loss(df, model_fn):
    N = len(df)
    total = None
    for _, row in df.iterrows():
        xi, yi = float(row["x"]), float(row["y"])
        term = (model_fn(xi) - yi) ** 2
        total = term if total is None else total + term
    return total * (1.0 / N)

L_linear = build_loss(df_linear, lambda xi: w * xi)
L_exp    = build_loss(df_exp,    lambda xi: exp(w * xi))
L_sin    = build_loss(df_sin,    lambda xi: sin(w * xi))
```

**Clave de diseño:** cada `xi` y `yi` son floats de Python; los operadores de `Function.__mul__`, `__sub__`, `__pow__` los envuelven automáticamente en `Const(xi)` mediante `to_function()`. No se requiere importar `Const` explícitamente ni usar numpy en la definición del árbol.

El resultado es que `L_linear`, `L_exp` y `L_sin` son objetos `Function` completos: soportan `.eval(w_val)` y `.derivative()`, lo que permite pasarlos directamente a `descenso_gradiente` sin modificar el algoritmo.

### Parámetros del dataset

| Característica | Valor |
|---|---|
| Filas por dataset | 100 |
| Rango de $x$ | $[-9.94,\ 9.81]$ |
| Los tres datasets comparten los mismos $x_i$ |

---

## 3. Cómo se cumple la consigna

| Requisito de la consigna | Cumplimiento |
|---|---|
| Construir $L(w)$ simbólicamente para cada modelo | ✓ `build_loss` genera un árbol `Function` con 100 términos por suma |
| Graficar $L(w)$ para cada dataset | ✓ Gráfica de 3 subplots con `w` en rangos $[-3,3]$, $[-3,3]$, $[0,10]$ |
| Describir el comportamiento observado en la gráfica | ✓ Preguntas 4.3 (primera celda markdown) |
| Aplicar descenso por gradiente para encontrar $w$ óptimo | ✓ Tabla de experimentos con 3 modelos y múltiples lr |
| Indicar si converge al resultado esperado | ✓ Comparación con OLS analítico para modelo lineal; análisis de overflow para exp |
| Graficar datos originales y modelo ajustado | ✓ 3 subplots scatter + curva |
| Responder preguntas 4.3 | ✓ Dos celdas markdown de análisis |

---

## 4. Resultados

### 4.1. Forma de L(w)

| Modelo | Forma observada |
|--------|----------------|
| Lineal | Parábola convexa con mínimo en $w^* \approx 1.85$; en el rango graficado $[-3, 3]$ se observa principalmente la rama descendente porque el mínimo está cerca del borde derecho |
| Exponencial | Asimétrica: explota hacia la **izquierda** ($w = -3 \Rightarrow L \approx 10^{24}$) cuando $w < 0$ y datos con $x_i < 0$ hacen que $e^{wx_i}$ crezca sin límite; mínimo en $w^* \approx 0.21$ en zona esencialmente plana |
| Sinusoidal | Oscilatoria con múltiples valles; no convexa |

### 4.2. Optimización — tabla completa de experimentos

Los experimentos se realizaron con `max_iter=5000` y `tol=1e-6`. El modelo exponencial usa dos puntos de partida para ilustrar el problema de escala.

```
        Modelo       lr  w encontrado  L(w) final  Iters ¿Convergió?
        Lineal 0.001000       1.84798    25.24204    158           ✓
        Lineal 0.010000       1.84799    25.24204     13           ✓
        Lineal 0.050000           NaN         NaN    406           ✗
  Exp. (w0=1.0) 0.000001           NaN         NaN      1           ✗
  Exp. (w0=1.0) 0.000010           NaN         NaN      1           ✗
  Exp. (w0=1.0) 0.000100           NaN         NaN      1           ✗
  Exp. (w0=0.1) 0.001000       0.21258     0.99438     10           ✓
  Exp. (w0=0.1) 0.010000           NaN         NaN      3           ✗
    Sinusoidal  0.001000       0.68980     1.19134    422           ✓
    Sinusoidal  0.010000       0.68977     1.19134     48           ✓
    Sinusoidal  0.050000       0.68977     1.19134     16           ✓

w* analítico (OLS, modelo lineal) = 1.84799
```

### 4.3. Verificación del modelo lineal

El valor encontrado por descenso ($w^* = 1.84799$) coincide con la solución analítica de mínimos cuadrados ordinarios ($w^*_{\text{OLS}} = 1.84799$) con error relativo $< 10^{-5}$. Esto confirma que la implementación es correcta para el caso convexo.

### 4.4. Análisis del overflow en el modelo exponencial

El dataset tiene $x_{\max} = 9.81$. En $w_0 = 1.0$:

$$e^{w_0 \cdot x_{\max}} = e^{9.81} \approx 18{,}138$$

El gradiente $L'(1)$ contiene factores de la forma $(e^{x_i} - y_i) \cdot x_i \cdot e^{x_i}$ para cada $i$. El término dominante ($x \approx 9.81$) vale aproximadamente:

$$2 \cdot (18138 - y) \cdot 9.81 \cdot 18138 \approx 6.5 \times 10^9 \quad \text{(por fila)}$$

La suma de los 100 términos y la división por $N$ producen un gradiente efectivo del orden de $10^8$. **Incluso con lr = $10^{-6}$, el primer paso es de magnitud $\approx 100$**, lo que desplaza $w$ a valores muy negativos donde los $x_i < 0$ generan nuevas explosiones de $e^{wx}$. El resultado es NaN desde la primera iteración para todo lr ensayado.

**Solución:** desde $w_0 = 0.1$, el factor más grande es $e^{0.1 \cdot 9.81} \approx 2.67$, y el gradiente es del orden de 10. Con lr = $10^{-3}$ el método converge en **10 iteraciones** a $w^* = 0.21258$.

### 4.5. Valores óptimos encontrados

| Modelo | $w^*$ | $L(w^*)$ | Método de partida |
|--------|--------|-----------|-------------------|
| Lineal | 1.84799 | 25.24204 | $w_0 = 1.0$, lr = $10^{-2}$ |
| Exponencial | 0.21258 | 0.99438 | $w_0 = 0.1$, lr = $10^{-3}$ |
| Sinusoidal | 0.68977 | 1.19134 | $w_0 = 1.0$, lr = $10^{-2}$ |

---

## 5. Respuestas a las preguntas de la consigna (sección 4.3)

### ¿Cómo cambia la forma de L(w) según el modelo?

**Lineal — convexa, cuadrática.**
$L_{\text{lin}}(w)$ es un polinomio de grado 2 en $w$ con coeficiente positivo. Es una parábola perfectamente suave y simétrica, sin extremos locales adicionales ni cambios de curvatura. Es la geometría de pérdida más favorable para optimización.

**Exponencial — unimodal, pero asimétrica.**
$L_{\text{exp}}(w)$ tiene un único mínimo en el rango relevante, pero es muy asimétrica: la rama derecha ($w > w^*$) crece explosivamente (por $e^{wx}$ con $x_{\max} = 9.81$), mientras que la rama izquierda ($w < w^*$) es comparativamente plana. Esta asimetría hace que el gradiente tenga magnitudes radicalmente distintas en distintas regiones, dificultando la elección de un lr fijo.

**Sinusoidal — no convexa, oscilatoria.**
$L_{\text{sin}}(w)$ hereda la periodicidad de $\sin(wx)$: distintos valores de $w$ producen frecuencias similares y pérdidas similares, generando múltiples valles. En el rango $w \in [0, 10]$ se observan varios mínimos locales. La función no es convexa.

---

### ¿Se observa un único mínimo en todos los casos? ¿Qué implica para la confiabilidad de descenso por gradiente?

| Modelo | Unicidad del mínimo | Confiabilidad del descenso |
|--------|---------------------|---------------------------|
| Lineal | **Sí**, único global | Siempre converge al óptimo global (con lr adecuado) |
| Exponencial | **Sí**, único en rango relevante | Converge al óptimo global si se elige $w_0$ apropiado |
| Sinusoidal | **No**, múltiples locales | El resultado depende de $w_0$; no garantiza el óptimo global |

**Implicación general:** descenso por gradiente es un optimizador *local*: solo garantiza encontrar un punto donde $L'(w) \approx 0$. En el modelo lineal esto no es un problema porque el único punto crítico es el mínimo global. En el modelo sinusoidal, distintos $w_0$ pueden llevar a distintos mínimos locales con distintos valores de $L(w)$. Para encontrar el global se necesitarían estrategias adicionales: múltiples reinicios aleatorios, recocido simulado, o restricción del rango de búsqueda mediante conocimiento del dominio.

---

### ¿Qué diferencias se observan en la convergencia? ¿Cuál fue más fácil de optimizar y por qué?

**Lineal — el más fácil.**
La curvatura es constante e igual a $\frac{2}{N}\sum x_i^2$, lo que garantiza convergencia lineal para cualquier $\text{lr} < \frac{N}{\sum x_i^2}$. El gradiente crece proporcionalmente a la distancia al mínimo: lejos, pasos grandes; cerca, pasos pequeños. No hay zonas problemáticas. Converge en 13 iteraciones con lr = $10^{-2}$.

**Sinusoidal — dificultad intermedia.**
El gradiente está globalmente acotado por $\frac{2}{N}\sum|x_i| \leq 2 x_{\max} \approx 20$, lo que elimina los problemas de escala. La complejidad reside en la no convexidad: distintos lr llevan a distintos mínimos locales (en este experimento, los tres lr convergen al mismo mínimo en $w \approx 0.690$, lo que sugiere que desde $w_0 = 1.0$ la cuenca de atracción de ese mínimo es amplia). La velocidad varía con el lr: de 422 iters (lr=$10^{-3}$) a 16 iters (lr=$5 \cdot 10^{-2}$).

**Exponencial — el más difícil.**
La asimetría extrema del gradiente impide la convergencia desde $w_0 = 1.0$ para todo lr ensayado. El problema es estructural: el rango de $x$ ($[-9.94, 9.81]$) combinado con la función $e^{wx}$ produce escalas de gradiente que difieren en órdenes de magnitud según la región de $w$. Un lr que evita la explosión inicial es demasiado pequeño para moverse en la zona plana y viceversa. La solución fue elegir $w_0 = 0.1$, próximo al óptimo real, donde las escalas son uniformes. Desde ahí converge en solo 10 iteraciones con lr = $10^{-3}$.

**Conclusión:** la dificultad de optimización no la determina únicamente la no convexidad, sino también la escala del gradiente en el rango de trabajo. El modelo exponencial, pese a tener un único mínimo global en el rango relevante, es el más difícil de optimizar por sus gradientes extremadamente no uniformes.
