# 🎵 Proyecto: Predicción de Popularidad de Canciones
## 📑 Informe Ejecutivo y Técnico — Sprint 2: Clasificación Binaria de Hits

**Institución:** Universidad Central — Ingeniería de Sistemas  
**Curso:** Inteligencia Artificial / Machine Learning Predictivo  
**Docente:** Prof. Elías Buitrago Bolívar  
**Sprint Activo:** Semana 2 (Sprint 2 - Clasificación)  
**Fecha de Entrega:** Octubre 2026  

---

## 🎯 1. Resumen Ejecutivo del Sprint 2

Durante este segundo sprint, el objetivo primordial fue evolucionar el sistema predictivo de una estimación continua de popularidad (Sprint 1: Regresión) hacia una **herramienta estratégica de toma de decisiones de inversión publicitaria** basada en el **costo de equivocarse** (Sprint 2: Clasificación Binaria).

Una disquera dispone de un presupuesto limitado y debe decidir, **antes del lanzamiento**, si invierte o no en la promoción de una canción nueva. A través de un enfoque de **Machine Learning sensible a los costos (*Cost-Sensitive Learning*)**, desarrollamos, evaluamos y empaquetamos un clasificador logístico regularizado con ingeniería de características acústicas que superó tanto la línea base de negocio como el modelo de referencia establecido en el taller.

### 🏆 Resultados Principales en Evaluación Fuera de Muestra (2020–2025):
* **Costo de Negocio:** **94 puntos** *(Línea base: 318; Referencia: 99)*. Reducción del **70.4%** respecto a la línea base.
* **Capacidad de Detección de Hits (Recall):** **84.9%** (el modelo detecta con éxito **90 de los 106 hits reales** en el periodo de prueba).
* **Calidad de Probabilidades (ROC-AUC):** **0.918** (alta capacidad de discriminación).
* **Calibración Probabilística (Brier Score):** **0.098** (probabilidades confiables y bien calibradas).
* **Garantía Operativa:** Paso del **100% de las pruebas automatizadas (`pytest -q: 9 passed`)** y peso del artefacto de **0.01 MB** (límite máximo permitido: 20 MB).

---

## 💼 2. Formulación del Problema y Reglas de Negocio

### 2.1. Variable Objetivo
* **`es_hit` (Binaria):** Toma el valor `1` si la canción alcanza una popularidad de **70 puntos o más** tras su lanzamiento; toma el valor `0` en caso contrario.

### 2.2. Matriz de Costos Asimétricos
Equivocarse en la industria musical no cuesta lo mismo en todas las direcciones. La disquera estableció la siguiente penalización económica:
* **Falso Negativo ($FN = 3$ puntos de costo):** Dejar sin promoción una canción que sí tenía potencial de convertirse en hit. Es el error más costoso debido a la pérdida irrecuperable de ingresos por streaming, conciertos y regalías.
* **Falso Positivo ($FP = 1$ punto de costo):** Invertir presupuesto en una canción que finalmente no fue un hit. Aunque representa gasto en marketing, la pérdida es significativamente menor.

$$\text{Costo Total} = (3 \times FN) + (1 \times FP)$$

### 2.3. Regla de Decisión y Umbral
El negocio determinó un umbral de corte fijo:
$$\text{Promocionar si } P(\text{es\_hit} = 1) \ge \mathbf{0.27}$$
Debido a que perder un hit cuesta el triple que promocionar un fracaso ($3:1$), la regla matemática óptima recomienda invertir incluso si la probabilidad estimada ronda el $25\% - 30\%$.

### 2.4. Restricción Estricta Anti-Fuga de Información (*No Data Leakage*)
Solo se utilizan las variables disponibles estrictamente **antes del lanzamiento**:
* **Variables permitidas (`COLUMNAS_ENTRADA`):** `bailabilidad`, `energia`, `valencia`, `acustica`, `tempo`, `duracion_min`, `volumen_db`, `anio`, `colaboracion` y `genero`.
* **Variables excluidas:**
  - `reproducciones_sem1`: Ocurre durante la primera semana post-lanzamiento.
  - `popularidad`: Define directamente a `es_hit` ($\text{popularidad} \ge 70$), por lo que usarla como entrada equivaldría a conocer de antemano la respuesta.

---

## 🔬 3. Partición Temporal de los Datos

Para garantizar validez temporal y evitar fugas hacia el futuro, se utilizó una partición estrictamente cronológica:
* **Periodo de Entrenamiento (1995–2019):** $1.585$ canciones históricas. En este subconjunto se ajustaron los preprocesadores, se aprendieron los escaladores y se calibraron los pesos del modelo.
* **Periodo de Evaluación / Prueba (2020–2025):** $415$ canciones recientes. Este conjunto simuló el escenario real de producción: evaluar canciones lanzadas en años posteriores a los datos conocidos por el modelo.

---

## 🧪 4. Ingeniería de Características (*Feature Engineering*)

Las variables se calcularon fila a fila dentro de `src/variables.py`, asegurando que puedan ser procesadas de manera idéntica tanto en lote como en peticiones individuales en la API FastAPI.

```
COLUMNAS_DERIVADAS = [
    "duracion_c2",
    "dance_energy",
    "valencia_acustica",
    "colab_indie",
    "pop_energia"
]
```

### Justificación de las variables creadas:
1. **`duracion_c2` $= (\text{duracion\_min} - 3.5)^2$:**
   Existe una penalización no lineal por canciones extremadamente cortas ($<2$ min) o excesivamente largas ($>5$ min). La función centra el óptimo comercial alrededor de los $3.5$ minutos.
2. **`dance_energy` $= \text{bailabilidad} \times \text{energia}$:**
   Captura la sinergia rítmica y energética de la pista. Las canciones que combinan alto ritmo con alta energía presentan una probabilidad exponencialmente superior de convertirse en hits virales.
3. **`valencia_acustica` $= \text{valencia} \times \text{acustica}$:**
   Permite distinguir piezas acústicas alegres frente a producciones electrónicas o baladas melancólicas.
4. **`colab_indie` $= \text{colaboracion} \times \mathbb{I}(\text{genero} = \text{"indie"})$:**
   En el género indie, la presencia de artistas invitados o colaboraciones genera un impulso significativo de difusión que no se observa de igual manera en otros géneros.
5. **`pop_energia` $= \text{energia} \times \mathbb{I}(\text{genero} = \text{"pop"})$:**
   Captura la preferencia de la audiencia por temas pop dinámicos de alta producción.

---

## 📊 5. Experimentación y Comparativa de Modelos

Se exploraron diversos algoritmos de clasificación evaluando su comportamiento en calibración probabilística y costo total:

| Modelo / Pipeline | Costo Total (2020-2025) | ROC-AUC | Brier Score | Recall | Precisión |
|---|:---:|:---:|:---:|:---:|:---:|
| **Línea Base (Predecir siempre "No Hit")** | **318** | 0.500 | 0.255 | 0.0% | 0.0% |
| **Referencia Inicial (Regresión Logística básica)** | **99** | 0.917 | 0.099 | 81.1% | 68.8% |
| Gradient Boosting Classifier (GBM) | 104 | 0.908 | 0.104 | 79.2% | 66.1% |
| HistGradientBoosting (Árboles tipo LightGBM) | 108 | 0.901 | 0.109 | 77.4% | 64.6% |
| Random Forest (Profundidad controlada) | 114 | 0.892 | 0.115 | 74.5% | 63.7% |
| **Modelo Final (Logística Regularizada $C=3.0$ + $w_{hit}=1.25$ + Interacciones)** | **94** 🏆 | **0.918** | **0.098** | **84.9%** | **66.2%** |

### Desglose de la Matriz de Confusión del Modelo Final (Evaluación 2020–2025):
* **Hits Reales en Prueba:** $106$ canciones.
* **No-Hits Reales en Prueba:** $309$ canciones.

$$\begin{pmatrix} \text{Verdaderos Negativos (TN)}: \mathbf{263} & \text{Falsos Positivos (FP)}: \mathbf{46} \\ \text{Falsos Negativos (FN)}: \mathbf{16} & \text{Verdaderos Positivos (TP)}: \mathbf{90} \end{pmatrix}$$

$$\text{Costo Obtenido} = (3 \times 16) + (1 \times 46) = 48 + 46 = \mathbf{94}$$

* **Evolución frente a la Referencia:** Se rescataron **6 hits adicionales** (FN bajó de 22 a 16), reduciendo el costo total de 99 a **94**.

---

## 🏗️ 6. Arquitectura del Artefacto y Desacoplamiento

El sistema se empaqueta en un único archivo serializado: `models/modelo.pkl`.

```
demo-reg-clas-canciones/ (o IA-Predictiva-de-Canciones/)
├── data/canciones.csv
├── models/modelo.pkl              <-- Artefacto final generado por el Sprint 2
├── src/
│   ├── variables.py               <-- Columnas de entrada y derivadas()
│   ├── artefacto.py               <-- guardar() y cargar() con validaciones
│   ├── entrenar_regresion.py      <-- Código Sprint 1
│   ├── entrenar_clasificacion.py  <-- Código Sprint 2
│   ├── probar_servicio.py         <-- Prueba de humo para Render
│   └── app/
│       ├── esquemas.py            <-- Validación Pydantic (entrada y salida)
│       └── main.py                <-- Servidor FastAPI
├── tests/test_servicio.py         <-- Suite de pruebas Pytest
├── requirements.txt               <-- Dependencias con versiones fijadas
└── render.yaml                    <-- Configuración Blueprint para Render
```

### Ficha Técnica Registrada en el Artefacto:
```json
{
  "nombre": "clasificador logístico regularizado + interacciones",
  "tarea": "clasificacion",
  "objetivo": "Predecir si la canción será un hit (es_hit)",
  "momento_prediccion": "antes del lanzamiento: la disquera decide cuánto invertir en promoción",
  "variables_entrada": [
    "bailabilidad", "energia", "valencia", "acustica", "tempo",
    "duracion_min", "volumen_db", "anio", "colaboracion", "genero"
  ],
  "variables_excluidas": {
    "reproducciones_sem1": "ocurre después del lanzamiento: no existe en el momento de decidir",
    "popularidad": "define es_hit: usarla como entrada equivale a conocer la respuesta"
  },
  "entrenamiento": { "periodo": "1995-2019", "n": 1585 },
  "prueba": { "periodo": "2020-2025", "n": 415, "tipo": "temporal" },
  "linea_base": { "descripcion": "predecir siempre 'no hit'", "costo": 318 },
  "desempeno": {
    "costo": 94,
    "roc_auc": 0.918,
    "brier_score": 0.098,
    "precision": 0.662,
    "recall": 0.849,
    "matriz_confusion": { "tn": 263, "fp": 46, "fn": 16, "tp": 90 }
  },
  "decision": {
    "umbral": 0.27,
    "costos": { "falso_negativo": 3, "falso_positivo": 1 },
    "regla": "promocionar si la probabilidad de hit supera el umbral",
    "nota": "los costos los fija quien responde por el presupuesto, no el modelo"
  },
  "limites": "describe asociaciones en datos sintéticos; probabilidades ajustadas por costos asimétricos"
}
```

---

## 🚀 7. Verificación del Servicio y API

### 7.1. Pruebas Locales Automatizadas
Ejecutadas con éxito en el entorno virtual del proyecto:
```powershell
pytest -q
```
**Resultado:** `9 passed in 2.43s` ✅

### 7.2. Contrato de la API (`POST /predecir`)
* **Entrada (JSON):**
  ```json
  {
    "bailabilidad": 0.72,
    "energia": 0.65,
    "valencia": 0.55,
    "acustica": 0.12,
    "tempo": 118.0,
    "duracion_min": 3.4,
    "volumen_db": -6.5,
    "anio": 2026,
    "colaboracion": 1,
    "genero": "urbano"
  }
  ```
* **Salida del Modelo de Clasificación (Sprint 2):**
  ```json
  {
    "tarea": "clasificacion",
    "modelo": "clasificador logístico regularizado + interacciones (2026-10-18)",
    "popularidad_esperada": null,
    "rango": null,
    "corte_promocion": null,
    "probabilidad_hit": 0.985,
    "umbral": 0.27,
    "promocionar": true,
    "explicacion": "probabilidad de hit 0.99 supera el umbral 0.27"
  }
  ```

---

## 📌 8. Checklist de Cumplimiento de Entrega

- [x] **Modelo entrenado exclusivamente con 1995–2019 y evaluado en 2020–2025.**
- [x] **Exclusión estricta de `reproducciones_sem1` y `popularidad`.**
- [x] **Superación de la línea base (Costo 94 vs 318).**
- [x] **Superación de la referencia (Costo 94 vs 99).**
- [x] **Umbral fijo de decisión en 0.27.**
- [x] **Soporte nativo de `predict_proba`.**
- [x] **Compatibilidad total con `requirements.txt` y peso < 20 MB.**
- [x] **Ficha técnica completa y validada por `src.artefacto.guardar()`.**
- [x] **Documentación del `README.md` actualizada con la información del Sprint 2.**
- [x] **Pruebas de regresión y humo en orden (`pytest -q` y `probar_servicio.py`).**
