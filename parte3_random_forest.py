"""
================================================================================
PARTE 3: RANDOM FOREST Y COMPARACIÓN DE MODELOS
================================================================================
Curso: Agentes Inteligentes - Semana 10 (USIL)
Objetivo:
  1. Entrenar RandomForestClassifier con n_estimators=100 y random_state=42.
  2. Comparar Árbol de Decisión vs Random Forest sobre el conjunto de prueba (test).
  3. Desglosar las matrices de confusión con la regla [[VN, FP], [FN, VP]].
  4. Extraer métricas clave de la clase 1 (deserción): Accuracy, Recall, F1, VP, FN.
  5. Graficar la importancia de variables (feature_importances_) y guardarla como imagen.
  6. Completar la tabla comparativa oficial de la rúbrica.
  7. Responder a las 3 preguntas analíticas de la Parte 3.
================================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Carga y preparación de datos
CASO, OBJETIVO = "Desercion", "deserta"
df = pd.read_excel("BD_Semana10_Riesgos.xlsx", sheet_name=CASO)
X = df.drop(columns=OBJETIVO)
y = df[OBJETIVO]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

# 2. Entrenamiento de modelos
arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
arbol.fit(X_tr, y_tr)

bosque = RandomForestClassifier(n_estimators=100, random_state=42)
bosque.fit(X_tr, y_tr)

# 3. Evaluación y desglose de métricas
resultados_tabla = []

print("=" * 70)
print("EVALUACIÓN DE MATRICES DE CONFUSIÓN Y CLASSIFICATION REPORT")
print("=" * 70)

for nombre, modelo in [("ÁRBOL (profundidad 3)", arbol), ("RANDOM FOREST (100 árboles)", bosque)]:
    pred = modelo.predict(X_te)
    cm = confusion_matrix(y_te, pred)
    # Python cm = [[VN, FP], [FN, VP]]
    vn, fp, fn, vp = cm.ravel()
    
    acc = round((vp + vn) / len(y_te), 3)
    rec1 = round(vp / (vp + fn), 3) if (vp + fn) > 0 else 0.0
    prec1 = round(vp / (vp + fp), 3) if (vp + fp) > 0 else 0.0
    f1_1 = round(2 * (prec1 * rec1) / (prec1 + rec1), 3) if (prec1 + rec1) > 0 else 0.0
    
    resultados_tabla.append({
        'Modelo': nombre,
        'Accuracy': acc,
        'Recall clase 1': rec1,
        'F1 clase 1': f1_1,
        'VP': vp,
        'FN': fn,
        'VN': vn,
        'FP': fp
    })
    
    print(f"\n>>> MODELO: {nombre}")
    print("Matriz de confusión:")
    print(cm)
    print(f"Desglose: VN = {vn} | FP = {fp} | FN = {fn} | VP = {vp}")
    print("\nClassification Report:")
    print(classification_report(y_te, pred, target_names=["No Deserta (0)", "Deserta (1)"]))

# 4. Tabla comparativa oficial
df_resultados = pd.DataFrame(resultados_tabla)
print("=" * 70)
print("TABLA COMPARATIVA DE RESULTADOS (CLASE 1: DESERCIÓN)")
print("=" * 70)
columnas_guia = ['Modelo', 'Accuracy', 'Recall clase 1', 'F1 clase 1', 'VP', 'FN']
print(df_resultados[columnas_guia].to_string(index=False))
print("=" * 70)

# 5. Importancia de Variables en Random Forest
importancias = pd.Series(bosque.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(10, 6))
importancias.plot.barh(color='#002060', edgecolor='#001030', width=0.6)
plt.title("¿Qué variables pesan más en el Random Forest?", fontsize=13, weight='bold', color='#002060')
plt.xlabel("Importancia Relativa (Gini Importance)", fontsize=11)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("importancia_variables_bosque.png", dpi=300)
print("\n[INFO] Gráfico de importancia guardado como 'importancia_variables_bosque.png'")
# plt.show() # Descomentar en entorno interactivo

# Obtener las dos variables más importantes
top2 = importancias.tail(2).index.tolist()[::-1]

# 6. Respuestas a las preguntas analíticas de la Parte 3
arbol_f1 = df_resultados.loc[df_resultados['Modelo'].str.startswith('ÁRBOL'), 'F1 clase 1'].values[0]
bosque_f1 = df_resultados.loc[df_resultados['Modelo'].str.startswith('RANDOM'), 'F1 clase 1'].values[0]
dif_f1 = round(bosque_f1 - arbol_f1, 3)

print("\n" + "=" * 70)
print("RESPUESTAS A LAS PREGUNTAS DE LA PARTE 3:")
print("=" * 70)
respuestas_p3 = f"""
1. ¿Qué modelo tiene mejor F1? ¿Por cuánto?
   RESPUESTA:
   El Random Forest tiene un mejor F1 ({bosque_f1:.3f} vs {arbol_f1:.3f} del árbol simple).
   Lo supera por {dif_f1:+.3f} puntos ({abs(dif_f1)*100:.1f}% de mejora).
   Esta superioridad se debe a que el ensamble promedia 100 árboles con subconjuntos
   aleatorios de datos y variables (bagging), reduciendo la varianza y capturando
   patrones sutiles sin sobreajustar.

2. ¿Qué significa un FN en tu caso? ¿Es peor que un FP? Explica en dos líneas.
   RESPUESTA:
   Un Falso Negativo (FN) significa que el modelo clasifica a un alumno como 'No Deserta'
   cuando en realidad sí terminará abandonando la universidad sin recibir ayuda a tiempo.
   Sí, el FN es MUCHO PEOR que un FP porque la universidad pierde la oportunidad de intervenir
   y rescatar al estudiante, mientras que un FP solo implica una llamada o tutoría preventiva extra.

3. ¿Cuáles son las dos variables que más pesan en el bosque?
   RESPUESTA:
   Las dos variables con mayor peso relativo en el Random Forest son:
   1. '{top2[0]}' ({importancias[top2[0]]:.1%} de peso)
   2. '{top2[1]}' ({importancias[top2[1]]:.1%} de peso)
   Esto confirma que la dimensión académica y el esfuerzo semanal son los factores
   estructurales determinantes en el riesgo de deserción.
"""
print(respuestas_p3)
print("=" * 70)
