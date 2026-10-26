# Plan de Sustentación, Guion de Exposición y Explicación Paso a Paso
## Actividad Práctica: Árbol de Decisión y Random Forest para Prevenir Riesgos
**Curso:** Agentes Inteligentes — Unidad 3 | **Universidad:** USIL  
**Docente:** Dr. José Alfredo Herrera Quispe | **Equipo:** Grupo 2  

---

## 🧭 1. Resumen de la Tarea y Archivos Creados en el Repositorio

Para resolver tu tarea y obtener la máxima calificación (20 puntos + 2 puntos del bono), se han creado archivos modulares e independientes para cada pregunta, más el cuaderno oficial y el simulador web:

| Archivo | Qué contiene y para qué sirve |
| :--- | :--- |
| **`tarea_sem10_arboles_random_forest.ipynb`** | **El Entregable Oficial de la Universidad.** Cuaderno Jupyter completo, 100% ejecutado con todas las salidas, tablas, gráficos de árboles y respuestas a la rúbrica. |
| **`BD_Semana10_Riesgos.xlsx`** | Base de datos oficial en Excel con las 8 hojas completas (`Desercion`, `ETS`, `Spam`, `Phishing`, `Extorsion`, `Phishing_mano`, `Diccionario`, `Fuentes`). |
| **`parte1_gini_manual.py`** | Script Python independiente de la **Parte 1** (Cálculo del Gini para los 14 correos, Gini de cada rama, Gini ponderado, árbol y reglas). |
| **`parte2_arbol_decision.py`** | Script Python independiente de la **Parte 2** (Árbol con `max_depth=3`, export_text, plot_tree y demostración de sobreajuste). |
| **`parte3_random_forest.py`** | Script Python independiente de la **Parte 3** (Random Forest con 100 árboles, matriz de confusión, tabla comparativa de métricas e importancia de variables). |
| **`parte4_conclusion.py`** | Script Python independiente de la **Parte 4** (Conclusión de 4 a 5 líneas redactada en lenguaje sencillo para no especialistas). |
| **`simulacion_interactiva.html` / `index.html`** | **Bono (+2 puntos):** Aplicación interactiva autocontenida (HTML, CSS y JS con Chart.js) lista para abrir en cualquier navegador o publicar en GitHub Pages. |
| **`PLAN_DE_SUSTENTACION_Y_EXPLICACION.md`** | **Este documento:** El guion exacto de texto que puedes leer y estudiar para tu sustentación oral. |

---

## 💡 Aclaración sobre los Lenguajes (¿Por qué Python y también HTML/JS?)
* **Python (`pandas`, `scikit-learn`, `matplotlib`):** Es el lenguaje que pidió el profesor para el núcleo del trabajo (Partes 0, 1, 2, 3 y 4) y el entregable oficial en formato Jupyter Notebook (`.ipynb`).
* **HTML, CSS y JavaScript (con Chart.js):** Se utilizan **exclusivamente para el Bono (+2 puntos)** de la "Simulación interactiva con IA", tal como pide el Anexo del profesor, para que puedas abrirlo en el navegador o publicarlo en GitHub Pages/Netlify.

---

# 🗣️ Guion de Lectura y Exposición Parte por Parte

---

## Parte 0 — Selección del Caso y Carga de Datos
> **Qué debes leer o decir:**  
> *"En la Parte 0, elegimos el **Caso de Deserción Universitaria** de la base de datos `BD_Semana10_Riesgos.xlsx`, inspirada en los datos del sistema Minedu-SIRIES. El objetivo es predecir si un estudiante abandonará sus estudios (`deserta`: 1 = Sí, 0 = No) en función de 5 variables: su promedio ponderado, su porcentaje de asistencia, la cantidad de cursos reprobados, los meses de atraso en pagos y sus horas semanales de estudio autónomo. El conjunto contiene 200 filas y 6 columnas."*

---

## Parte 1 — Árbol a Mano con Índice Gini (14 Correos) — [4 Puntos]
> **Qué debes leer o decir:**  
> *"En la Parte 1 resolvimos el árbol a mano para clasificar 14 correos (7 eran phishing y 7 eran legítimos).  
> 1. Primero calculamos el **Gini de la raíz**: como las clases están empatadas 50/50, el Gini inicial es $1 - (0.5^2 + 0.5^2) = 0.5000$.  
> 2. Luego evaluamos el Gini de cada rama y el Gini ponderado para las 4 variables:  
>    - **Urgencia:** Gini ponderado = **0.1714**  
>    - **Remitente:** Gini ponderado = **0.4583**  
>    - **Adjunto:** Gini ponderado = **0.4898**  
>    - **Enlace corto:** Gini ponderado = **0.5000**  
> 3. Como **Urgencia** obtuvo el menor Gini ponderado (0.1714), es la variable más pura y se elige como **nodo raíz**.  
> 4. Al abrir las ramas de Urgencia:  
>    - Si la Urgencia es 'Alta', los 4 correos son Phishing (Gini = 0, hoja pura).  
>    - Si la Urgencia es 'Baja', los 5 correos son No Phishing (Gini = 0, hoja pura).  
>    - Si la Urgencia es 'Media', quedan 5 correos (3 Sí, 2 No). Al subdividir por **Remitente**, Desconocido separa 3 Sí y Conocido separa 2 No, obteniendo un Gini de 0.0000.  
> 5. Así obtuvimos las 4 reglas lógicas:  
>    - *SI Urgencia = "Alta" ENTONCES Phishing = "Sí"*  
>    - *SI Urgencia = "Baja" ENTONCES Phishing = "No"*  
>    - *SI Urgencia = "Media" Y Remitente = "Desconocido" ENTONCES Phishing = "Sí"*  
>    - *SI Urgencia = "Media" Y Remitente = "Conocido" ENTONCES Phishing = "No"*  
>    - *EN OTRO CASO ENTONCES Phishing = "No"* (regla preventiva por defecto)."*

---

## Parte 2 — Tu Árbol en Python y Sobreajuste — [5 Puntos]
> **Qué debes leer o decir:**  
> *"En la Parte 2 entrenamos el árbol en Python usando `scikit-learn`:  
> 1. Dividimos los datos en 70% entrenamiento (140 filas) y 30% prueba (60 filas) con `test_size=0.3`, `random_state=42` y estratificación.  
> 2. Entrenamos un `DecisionTreeClassifier` con profundidad controlada (`max_depth=3`). Obtuvo un 90.7% de acierto en entrenamiento y 80.0% en prueba.  
> 3. La variable que eligió como raíz fue **`cursos_reprobados`** (umbral $\le 2.5$) y luego **`promedio_ponderado`**, lo cual tiene total sentido lógico porque jalar 3 o más asignaturas en la universidad es el primer síntoma de abandono estudiantil.  
> 4. Dos reglas simples del árbol son:  
>    - *Regla 1:* SI reprobó más de 2 cursos y estudia $\le 23.8$ horas a la semana, ENTONCES abandonará la universidad (Deserta = Sí).  
>    - *Regla 2:* SI reprobó 2 o menos cursos y su promedio es mayor a 10.02, ENTONCES continuará estudiando (Deserta = No).  
> 5. Al comparar con el **árbol libre sin límite de profundidad**, este sacó 100% (1.0) en entrenamiento pero cayó a 78.3% en prueba. Esto demuestra el fenómeno de **sobreajuste (overfitting)**: el árbol sin podar memorizó el ruido y las peculiaridades de entrenamiento, perdiendo capacidad de generalizar con alumnos nuevos."*

---

## Parte 3 — Random Forest y Comparación de Modelos — [5 Puntos]
> **Qué debes leer o decir:**  
> *"En la Parte 3 implementamos un **Random Forest** con 100 árboles de decisión (`n_estimators=100`, `random_state=42`) y lo comparamos con el árbol simple:  
> 1. Desglosamos la matriz de confusión con el orden de Python `[[VN, FP], [FN, VP]]`.  
> 2. Al llenar la tabla comparativa oficial de la clase 1 (Deserción):  
>    - El **Árbol simple** obtuvo: Accuracy = **0.800**, Recall = **0.636**, F1 = **0.700**, VP = **14**, FN = **8**.  
>    - El **Random Forest** obtuvo: Accuracy = **0.833**, Recall = **0.727**, F1 = **0.762**, VP = **16**, FN = **6**.  
> 3. **¿Qué modelo tiene mejor F1?** El Random Forest supera al árbol por **+0.062 puntos (+6.2%)**. Esto ocurre porque al promediar 100 árboles entrenados con diferentes muestras y variables aleatorias (bagging), se reduce drásticamente la varianza y los errores individuales se compensan.  
> 4. **¿Qué significa un FN y por qué es peor que un FP?** Un Falso Negativo (FN) significa que el modelo predijo que el alumno NO desertaría, pero en la realidad sí abandonó la universidad. Es **mucho peor que un FP** porque el estudiante se retira sin recibir ninguna ayuda, mientras que un FP solo cuesta una llamada o cita de tutoría preventiva de más.  
> 5. **¿Cuáles son las dos variables que más pesan en el bosque?** Según `feature_importances_`, son **`promedio_ponderado`** (34.7%) y **`cursos_reprobados`** (31.3%), que juntas concentran más del 66% de la capacidad de decisión."*

---

## Parte 4 — Conclusión para No Especialistas — [2 Puntos]
> **Qué debes leer textualmente (el párrafo de 4 a 5 líneas):**  
> *"Para el equipo de tutoría universitaria, recomendamos implementar el modelo de **Random Forest (bosque de decisión)**, ya que demostró la mayor efectividad para detectar a los estudiantes en riesgo real (alcanzando un F1 del 76.2% y reduciendo los casos inadvertidos a solo 6). Los tutores deben vigilar prioritariamente el **promedio ponderado acumulado** (especialmente si desciende de 10) y la cantidad de **cursos reprobados** (2 o más asignaturas). Como acción concreta, proponemos activar un **sistema de alerta temprana en la semana 4 de cada ciclo** que agende citas de mentoría académica personalizada y facilidades de reprogramación de pensiones antes de que la deserción sea inevitable."*

---

# 🎓 Preguntas de Sustentación Oral (4 Puntos)

El docente te hará una o varias de estas 4 preguntas. Aquí tienes la respuesta exacta y directa para responder con seguridad:

### 1. ¿Por qué elegiste esa variable como raíz en la Parte 1?
> **Respuesta:**  
> *"Elegí **`Urgencia`** porque es la variable que minimiza el índice de impureza de Gini ponderado, alcanzando **0.1714** frente a los 0.5000 iniciales. Ninguna otra variable logró una reducción tan grande (`Remitente` dio 0.4583, `Adjunto` 0.4898 y `Enlace corto` 0.5000). Además, `Urgencia` dividió directamente 9 de los 14 correos en hojas 100% puras (Alta=Sí con Gini 0, y Baja=No con Gini 0)."*

### 2. Lee una regla de tu árbol y di qué acción tomarías.
> **Respuesta:**  
> *"La regla dice: **SI cursos_reprobados > 2.5 Y horas_estudio_semanal $\le 23.8$ ENTONCES Deserta = Sí**.  
> **Acción que tomaría:** Si detectamos a un estudiante con más de dos materias desaprobadas y pocas horas de estudio independiente, la dirección de carrera debe asignarle de inmediato un tutor o mentor par para enseñarle técnicas de estudio y organización del tiempo, además de evaluar una reducción de carga académica para que no se frustre y abandone."*

### 3. ¿Por qué un bosque suele acertar más que un solo árbol?
> **Respuesta:**  
> *"Porque un solo árbol de decisión sufre de **alta varianza**: es muy sensible a los datos y tiende a sobreajustarse memorizando el ruido. En cambio, el Random Forest crea una asamblea o bosque de cientos de árboles entrenados con subconjuntos aleatorios de datos (*bootstrap*) y subconjuntos aleatorios de variables. Al final, las predicciones se combinan por **votación mayoritaria**. Gracias a la ley de los grandes números y la diversidad del ensamble, los errores individuales de los árboles se cancelan, logrando un modelo mucho más robusto y que generaliza mejor ante datos no vistos."*

### 4. En tu caso, ¿qué error es más grave, FN o FP?
> **Respuesta:**  
> *"En deserción universitaria, el error más grave es el **Falso Negativo (FN)**. Si cometemos un FP, catalogamos en riesgo a un alumno que no planeaba retirarse, lo cual solo implica brindarle una tutoría preventiva de bajo costo que igual le suma. Pero un FN significa dejar en silencio a un alumno vulnerable que terminará abandonando la universidad, lo que constituye un daño irreparable para el futuro del estudiante y una pérdida de retención para la institución."*

---

# 🌐 Bono (+2 Puntos): Cómo Publicar la Simulación Interactiva

El profesor indica que puedes publicar la simulación gratis en **GitHub Pages** o **Netlify Drop**:

### Opción 1: En GitHub Pages (En 1 minuto)
1. Haz commit y push de tus archivos a tu repositorio de GitHub:
   ```bash
   git add .
   git commit -m "Entrega actividad Semana 10 con simulador interactivo"
   git push origin main
   ```
2. En GitHub, entra a tu repositorio: `tarea-sem-10-AGi-grupo-2`.
3. Ve a la pestaña **Settings** > en el menú izquierdo haz clic en **Pages**.
4. En **Branch**, selecciona `main` y la carpeta `/(root)`, luego haz clic en **Save**.
5. En 60 segundos, GitHub te dará un enlace público tipo:  
   `https://jussepe06.github.io/tarea-sem-10-AGi-grupo-2/`  
   (Como el archivo también se guardó como `index.html`, cargará automáticamente).

### Opción 2: En Netlify Drop (En 30 segundos)
1. Entra a [https://app.netlify.com/drop](https://app.netlify.com/drop).
2. Arrastra la carpeta de tu proyecto o el archivo `simulacion_interactiva.html`.
3. Netlify te dará un enlace web instantáneo gratuito para compartir en tu entrega.
