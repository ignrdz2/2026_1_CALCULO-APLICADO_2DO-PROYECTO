# Parte 2 — Implementación y verificación de métodos de optimización

## 1. Marco teórico

El problema de optimización en una dimensión consiste en encontrar un punto `x*` donde una función `f(x)` alcanza un máximo o mínimo local. La condición necesaria de primer orden establece que en todo extremo interior se cumple `f'(x*) = 0`. Por ello, los métodos implementados no operan directamente sobre `f`, sino sobre su derivada: buscan raíces de `f'`.

### Método de bisección

Dado un intervalo `[a, b]` tal que `f'(a)` y `f'(b)` tienen signos opuestos, por el Teorema del Valor Intermedio existe al menos una raíz de `f'` en ese intervalo. El método bisecta el intervalo iterativamente:

```
m = (a + b) / 2
si f'(a) · f'(m) < 0  →  b = m
si no                  →  a = m
```

La convergencia es **lineal**: el error se reduce a la mitad en cada iteración. Para una tolerancia `ε`, se necesitan aproximadamente `log₂((b−a)/ε)` iteraciones.

### Método de Newton (para optimización)

Aplica Newton-Raphson sobre `f'` para encontrar su raíz. La regla de actualización es:

```
x_{n+1} = x_n − f'(x_n) / f''(x_n)
```

La convergencia es **cuadrática** cerca de la solución: el número de dígitos correctos se duplica en cada iteración. Requiere que `f''(x_n) ≠ 0`.

### Descenso por gradiente

Actualiza `x` en la dirección contraria al gradiente de `f`, con una tasa de aprendizaje `lr` que controla el tamaño del paso:

```
x_{n+1} = x_n − lr · f'(x_n)
```

La convergencia es **lineal** y fuertemente dependiente de `lr`. Si `lr` es demasiado grande el método puede divergir; si es demasiado pequeño converge muy lentamente.

---

## 2. Qué se implementó y cómo

Los tres métodos se implementaron en el notebook `project.ipynb` como funciones Python que reciben objetos del sistema de diferenciación simbólica del proyecto (`abstractions/`). La interfaz de ese sistema expone dos operaciones sobre cualquier función:

- `f.eval(x)` — evaluación numérica en un punto
- `f.derivative()` — devuelve la derivada simbólica como otro objeto `Function`

Esto permite construir `f'` y `f''` **una sola vez** antes del loop y reutilizarlas en cada iteración, evitando reconstruir el árbol de derivadas en cada paso.

### `biseccion(f, a, b, tol=1e-6, max_iter=1000)`

```python
def biseccion(f, a, b, tol=1e-6, max_iter=1000):
    df = f.derivative()
    step = 0
    dfa = df.eval(a)

    while step < max_iter and (b - a) >= tol:
        m = (a + b) / 2
        dfm = df.eval(m)
        step += 1
        if abs(dfm) < tol:
            return m, step
        if dfa * dfm < 0:
            b = m
        else:
            a = m
            dfa = dfm

    return (a + b) / 2, step
```

**Decisiones de implementación:**

- `dfa` se evalúa fuera del loop y se actualiza solo cuando `a` cambia (no cuando cambia `b`), reduciendo evaluaciones de `df` a una por iteración.
- Se añade un chequeo `abs(dfm) < tol` antes de la comparación de signos. Esto resuelve el caso degenerado donde el punto medio es exactamente la raíz: el producto `f'(a) · f'(m) = 0` no es negativo y sin este chequeo el intervalo se desplaza en la dirección incorrecta.

### `newton(f, x0, tol=1e-6, max_iter=1000)`

```python
def newton(f, x0, tol=1e-6, max_iter=1000):
    df  = f.derivative()
    d2f = df.derivative()
    xn  = x0
    step = 0

    while step < max_iter:
        d2f_val = d2f.eval(xn)
        if abs(d2f_val) < 1e-12:
            break
        x_new = xn - df.eval(xn) / d2f_val
        step += 1
        if abs(x_new - xn) < tol:
            return x_new, step
        xn = x_new

    return xn, step
```

**Decisiones de implementación:**

- `df` y `d2f` se construyen una sola vez.
- Si `|f''(xn)| < 1e-12` se interrumpe el loop: el denominador sería numéricamente nulo (punto de inflexión o función plana), y la actualización produciría un valor sin sentido.
- El umbral de la segunda derivada (`1e-12`) es mucho menor que la tolerancia de convergencia (`1e-6`) para no confundirlos.

### `descenso_gradiente(f, x0, lr, tol=1e-6, max_iter=1000)`

```python
def descenso_gradiente(f, x0, lr, tol=1e-6, max_iter=1000):
    df = f.derivative()
    xn = x0
    step = 0

    while step < max_iter:
        x_new = xn - lr * df.eval(xn)
        step += 1
        if abs(x_new - xn) < tol:
            return x_new, step
        xn = x_new

    return xn, step
```

**Decisiones de implementación:**

- Es el más simple de los tres: una sola línea de lógica por iteración.
- No tiene casos degenerados propios: siempre puede dar un paso (aunque con `lr` inadecuado puede divergir).

---

## 3. Cómo se cumple la consigna

| Requisito de la consigna                                  | Cumplimiento                                                                |
| --------------------------------------------------------- | --------------------------------------------------------------------------- |
| Implementar bisección, Newton y descenso por gradiente    | ✓ Los tres implementados en la celda de métodos del notebook                |
| Recibir condiciones iniciales apropiadas para cada método | ✓ Bisección recibe `[a,b]`; Newton recibe `x0`; descenso recibe `x0` y `lr` |
| Incluir criterio de parada por tolerancia                 | ✓ `\|b−a\| < tol` (bisección), `\|x_new−xn\| < tol` (Newton y descenso)     |
| Incluir criterio de parada por iteraciones máximas        | ✓ `while step < max_iter` en los tres                                       |
| Devolver el punto obtenido y la cantidad de iteraciones   | ✓ Todos devuelven `(xn, step)`                                              |
| Verificar sobre funciones con extremo identificable       | ✓ Verificado sobre `f1`, `f2`, `f3` con grilla 3×3 de plots                 |

---

## 4. Resultados de verificación

Las funciones de prueba y sus extremos esperados:

| Función                   | Extremo teórico                      | xn esperado | f(xn) esperado |
| ------------------------- | ------------------------------------ | ----------- | -------------- |
| `f1 = x²`                 | mínimo global                        | x = 0       | f = 0          |
| `f2 = (x+0.5)³ − (x+0.5)` | mínimo local (con `[−1,1]` / `x0=1`) | x ≈ 0.0774  | f ≈ −0.3849    |
| `f3 = −cos(x)`            | mínimo local                         | x = 0       | f = −1         |

Resultados obtenidos (tolerancia `1e-6`):

| Función           | Bisección `[a,b]=[−1,1]`    | Newton `x0=1`              | Descenso `x0=1, lr=1e-2`     |
| ----------------- | --------------------------- | -------------------------- | ---------------------------- |
| `f1 = x²`         | xn = 0.000000 **(1 iter)**  | xn = 0.000000 **(2 iter)** | xn ≈ 0.000048 **(492 iter)** |
| `f2 = (x+0.5)³−…` | xn = 0.077350 **(21 iter)** | xn = 0.077350 **(6 iter)** | xn ≈ 0.077378 **(278 iter)** |
| `f3 = −cos(x)`    | xn = 0.000000 **(1 iter)**  | xn = 0.000000 **(5 iter)** | xn ≈ 0.000098 **(927 iter)** |

Todos los métodos convergen al extremo correcto en los tres casos. Newton es consistentemente el más rápido; bisección es eficiente cuando la raíz cae exactamente en el punto medio; descenso por gradiente es el más lento pero no requiere ni intervalo ni segunda derivada.

---

## 5. Respuestas a las preguntas de la consigna (sección 2.3)

### ¿Pueden los métodos diferenciar un máximo de un mínimo?

No directamente. Los tres métodos hallan puntos donde `f'(x) = 0`, sin distinguir por sí solos si ese punto es un máximo, un mínimo o un punto de inflexión. El algoritmo que converge al mínimo o al máximo depende únicamente de las condiciones iniciales, no del método en sí.

En `f2 = (x+0.5)³ − (x+0.5)`, la derivada `f2'(x) = 3(x+0.5)² − 1` tiene dos raíces:

- `x ≈ 0.077` → mínimo local (`f2''(0.077) > 0`)
- `x ≈ −1.077` → máximo local (`f2''(−1.077) < 0`)

Con `[a,b] = [−1, 1]` y `x0 = 1`, todos los métodos convergen al mínimo. Con `[a,b] = [−2, −0.5]` o `x0 = −2` convergerían al máximo. El algoritmo no distingue.

**Para identificar si `xn` es máximo o mínimo**, se aplica el criterio de la segunda derivada usando el sistema simbólico:

```python
clasificacion = f.derivative().derivative().eval(xn)
# > 0  →  mínimo local
# < 0  →  máximo local
# ≈ 0  →  test inconcluso
```

### ¿Qué ventajas y desventajas tiene cada método?

**Bisección**

- _Ventajas_: convergencia garantizada si se cumple la condición de signos opuestos; robusto, sin singularidades.
- _Desventajas_: convergencia lineal (lenta); exige conocer un intervalo `[a,b]` con `f'(a)·f'(b) < 0`, lo que requiere conocimiento previo de la función; solo puede aislar un extremo por intervalo.

**Newton**

- _Ventajas_: convergencia cuadrática (muy rápida); pocas iteraciones incluso para tolerancias estrictas.
- _Desventajas_: requiere `f''`; falla si `f''(xn) ≈ 0`; puede divergir o saltar a otro extremo dependiendo del punto inicial; sin garantía de convergencia global.

**Descenso por gradiente**

- _Ventajas_: requiere solo `f'`; solo necesita un punto inicial; es la base de los optimizadores modernos en aprendizaje automático.
- _Desventajas_: convergencia lineal y lenta; muy sensible a `lr`; puede quedar atrapado en extremos locales; el número de iteraciones puede ser órdenes de magnitud mayor que Newton.

![1780488115037](image/informe_parte2/1780488115037.png)
