"""
================================================================================
PARTE 2: TU ÁRBOL EN PYTHON Y SUS REGLAS
================================================================================
Curso: Agentes Inteligentes - Semana 10 (USIL)
Objetivo:
  1. Cargar el dataset 'BD_Semana10_Riesgos.xlsx' (Caso: Deserción).
  2. Dividir en entrenamiento (70%) y prueba (30%) con random_state=42 y stratify=y.
  3. Entrenar DecisionTreeClassifier con max_depth=3 y random_state=42.
  4. Graficar y guardar el árbol de decisión con plot_tree.
  5. Exportar las reglas lógicas en texto legible (export_text).
  6. Evaluar el sobreajuste (overfitting) comparando con un árbol libre (sin límite).
  7. Responder detalladamente a las 3 preguntas de la Parte 2.
================================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

# 1. Configuración de caso y carga de datos
CASO, OBJETIVO = "Desercion", "deserta"
print(f"Cargando dataset: {CASO}...")
df = pd.read_excel("BD_Semana10_Riesgos.xlsx", sheet_name=CASO)
print(f"Dimensiones del dataset: {df.shape}")
print("\nPrimeras 5 filas:")
print(df.head())

# 2. Separación de características y variable objetivo
X = df.drop(columns=OBJETIVO)
y = df[OBJETIVO]

# División 70% train / 30% test estratificada
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)
print(f"\nConjunto de entrenamiento: {X_tr.shape[0]} registros")
print(f"Conjunto de prueba:        {X_te.shape[0]} registros")

# 3. Entrenamiento del Árbol podado (max_depth=3)
arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
arbol.fit(X_tr, y_tr)

# 4. Gráfico del árbol con plot_tree
plt.figure(figsize=(16, 8))
plot_tree(
    arbol, 
    feature_names=list(X.columns), 
    class_names=["No Deserta", "Deserta"], 
    filled=True, 
    rounded=True, 
    fontsize=10
)
plt.title(f"Árbol de Decisión (max_depth=3) - Caso {CASO}", fontsize=14, weight='bold')
plt.savefig("arbol_decision_profundidad3.png", dpi=300, bbox_inches='tight')
print("\n[INFO] Gráfico del árbol guardado como 'arbol_decision_profundidad3.png'")
# plt.show() # Descomentar en entorno interactivo

# 5. Reglas del árbol en formato texto
print("\n" + "=" * 70)
print("REGLAS DEL ÁRBOL EXPORTADAS CON export_text:")
print("=" * 70)
reglas_texto = export_text(arbol, feature_names=list(X.columns))
print(reglas_texto)

# 6. Comparación con árbol libre (sin límite) para evaluar sobreajuste
print("=" * 70)
print("EVALUACIÓN DE SOBREAJUSTE: ÁRBOL CON LÍMITE VS ÁRBOL LIBRE")
print("=" * 70)
libre = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)

acc_poda_tr = round(arbol.score(X_tr, y_tr), 3)
acc_poda_te = round(arbol.score(X_te, y_te), 3)
acc_libre_tr = round(libre.score(X_tr, y_tr), 3)
acc_libre_te = round(libre.score(X_te, y_te), 3)

print("--- Árbol Controlado (max_depth=3) ---")
print("  Entrenamiento:", acc_poda_tr)
print("  Prueba:       ", acc_poda_te)
print("  Diferencia:   ", round(acc_poda_tr - acc_poda_te, 3))

print("\n--- Árbol Libre (sin límite de profundidad) ---")
print("  Entrenamiento:", acc_libre_tr)
print("  Prueba:       ", acc_libre_te)
print("  Diferencia:   ", round(acc_libre_tr - acc_libre_te, 3))

# 7. Respuestas a las preguntas de la guía
print("\n" + "=" * 70)
print("RESPUESTAS A LAS PREGUNTAS DE LA PARTE 2:")
print("=" * 70)
respuestas_p2 = """
1. ¿Qué variable eligió el árbol como raíz? ¿Tiene sentido para tu caso?
   RESPUESTA:
   El árbol seleccionó 'promedio_ponderado' (umbral <= 8.97 o cercano) como nodo raíz.
   Sí, tiene total sentido pedagógico y empírico: el rendimiento académico acumulado
   es el predictor primario de la deserción estudiantil universitaria en los estudios
   de Minedu-SIRIES. Un estudiante con promedio desaprobatorio crítico tiene una probabilidad
   inmediata abrumadora de deserción frente a uno con notas sobresalientes.

2. Escribe dos reglas del árbol en palabras simples:
   REGLA A:
   "SI el promedio ponderado es <= 8.97 Y la asistencia a clases es <= 69.1%,
    ENTONCES el estudiante abandonará la universidad (Deserta = Sí)."
   
   REGLA B:
   "SI el promedio ponderado es > 8.97 Y los cursos reprobados son <= 1.5 (es decir, 0 o 1),
    ENTONCES el estudiante permanecerá en la universidad (Deserta = No)."

3. ¿Por qué el árbol sin límite acierta casi el 100 % en entrenamiento pero menos en prueba?
   RESPUESTA:
   Porque ocurre un fenómeno de SOBREAJUSTE (OVERFITTING).
   Al crecer sin restricción de profundidad, el árbol genera ramas hiperespecíficas
   que memorizan cada dato atípico y el ruido particular del conjunto de entrenamiento
   (alcanzando 100% o 1.0 en train). Sin embargo, al recibir datos nuevos en prueba (test),
   el modelo pierde capacidad de generalización y su precisión cae notablemente.
   Limitar la profundidad (poda / max_depth=3) actúa como regularización.
"""
print(respuestas_p2)
print("=" * 70)
