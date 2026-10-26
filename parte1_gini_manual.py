"""
================================================================================
PARTE 1: ÁRBOL A MANO CON ÍNDICE GINI (14 CORREOS DE PHISHING)
================================================================================
Curso: Agentes Inteligentes - Semana 10 (USIL)
Objetivo:
  1. Calcular el Gini inicial del conjunto de datos (raíz).
  2. Calcular el Gini de cada rama y el Gini ponderado para cada una de las 
     4 variables: Urgencia, Remitente, Adjunto y Enlace corto.
  3. Elegir la variable con el menor Gini ponderado como raíz del árbol.
  4. Continuar la partición en las ramas impuras hasta obtener ramas 100% puras.
  5. Extraer las reglas lógicas SI... ENTONCES con su regla EN OTRO CASO.
================================================================================
"""

import pandas as pd
import numpy as np

# Datos oficiales de los 14 correos
datos_correos = [
    [1, 'Alta', 'Desconocido', 'Sí', 'Sí', 'Sí'],
    [2, 'Alta', 'Desconocido', 'No', 'No', 'Sí'],
    [3, 'Alta', 'Conocido', 'Sí', 'No', 'Sí'],
    [4, 'Alta', 'Conocido', 'No', 'Sí', 'Sí'],
    [5, 'Media', 'Desconocido', 'Sí', 'Sí', 'Sí'],
    [6, 'Media', 'Desconocido', 'No', 'No', 'Sí'],
    [7, 'Media', 'Desconocido', 'Sí', 'No', 'Sí'],
    [8, 'Media', 'Conocido', 'No', 'Sí', 'No'],
    [9, 'Media', 'Conocido', 'Sí', 'No', 'No'],
    [10, 'Baja', 'Desconocido', 'No', 'Sí', 'No'],
    [11, 'Baja', 'Desconocido', 'Sí', 'No', 'No'],
    [12, 'Baja', 'Conocido', 'No', 'No', 'No'],
    [13, 'Baja', 'Conocido', 'Sí', 'Sí', 'No'],
    [14, 'Baja', 'Desconocido', 'No', 'No', 'No']
]

columnas = ['#', 'Urgencia', 'Remitente', 'Adjunto', 'Enlace_corto', 'Phishing']
df = pd.DataFrame(datos_correos, columns=columnas)

def calcular_gini(serie_objetivo):
    """Calcula Gini = 1 - sum(p_i^2) para clasificación binaria."""
    total = len(serie_objetivo)
    if total == 0:
        return 0.0
    p_si = (serie_objetivo == 'Sí').sum() / total
    p_no = (serie_objetivo == 'No').sum() / total
    return 1.0 - (p_si**2 + p_no**2)

print("=" * 70)
print("1. CÁLCULO DEL GINI INICIAL (RAÍZ DEL DATASET COMPLETO)")
print("=" * 70)
total_emails = len(df)
total_si = (df['Phishing'] == 'Sí').sum()
total_no = (df['Phishing'] == 'No').sum()
gini_raiz = calcular_gini(df['Phishing'])

print(f"Total de correos: {total_emails}")
print(f"Phishing = Sí: {total_si} ({total_si}/{total_emails} = {total_si/total_emails:.4f})")
print(f"Phishing = No: {total_no} ({total_no}/{total_emails} = {total_no/total_emails:.4f})")
print(f"Fórmula: Gini(Raíz) = 1 - [(p_Sí)^2 + (p_No)^2]")
print(f"Gini(Raíz) = 1 - [({total_si}/{total_emails})^2 + ({total_no}/{total_emails})^2]")
print(f"Gini(Raíz) = 1 - [0.5^2 + 0.5^2] = 1 - 0.50 = {gini_raiz:.4f}")
print("=" * 70)

print("\n" + "=" * 70)
print("2. EVALUACIÓN DE LAS 4 VARIABLES CANDIDATAS PARA EL NODO RAÍZ")
print("=" * 70)

variables = ['Urgencia', 'Remitente', 'Adjunto', 'Enlace_corto']
resultados_gini = {}

for var in variables:
    print(f"\n>>> Variable: {var}")
    gini_pond = 0.0
    for val, grupo in df.groupby(var):
        g = calcular_gini(grupo['Phishing'])
        cnt_si = (grupo['Phishing'] == 'Sí').sum()
        cnt_no = (grupo['Phishing'] == 'No').sum()
        peso = len(grupo) / total_emails
        gini_pond += peso * g
        print(f"   Rama '{val}': {len(grupo)} correos (Sí={cnt_si}, No={cnt_no}) -> Gini = 1 - [({cnt_si}/{len(grupo)})^2 + ({cnt_no}/{len(grupo)})^2] = {g:.4f}")
    
    resultados_gini[var] = gini_pond
    print(f"   => GINI PONDERADO de {var} = {gini_pond:.4f}")

variable_raiz = min(resultados_gini, key=resultados_gini.get)
print("\n" + "-" * 70)
print("RESUMEN DE GINI PONDERADO POR VARIABLE:")
for var, val in sorted(resultados_gini.items(), key=lambda x: x[1]):
    print(f"  * {var:15s}: {val:.4f}")

print(f"\nCONCLUSIÓN DEL PASO 1:")
print(f"La variable con MENOR Gini ponderado es '{variable_raiz}' ({resultados_gini[variable_raiz]:.4f}).")
print(f"Por lo tanto, la RAÍZ DEL ÁRBOL es: {variable_raiz}")
print("-" * 70)

print("\n" + "=" * 70)
print("3. ANÁLISIS DE LAS RAMAS DE LA VARIABLE RAÍZ ('Urgencia')")
print("=" * 70)
print("• Rama Urgencia = 'Alta':")
print("  Total = 4 correos (4 Sí, 0 No) -> Gini = 0.0000. ¡HOJA PURA!")
print("  Decisión: Phishing = Sí")

print("\n• Rama Urgencia = 'Baja':")
print("  Total = 5 correos (0 Sí, 5 No) -> Gini = 0.0000. ¡HOJA PURA!")
print("  Decisión: Phishing = No")

print("\n• Rama Urgencia = 'Media':")
print("  Total = 5 correos (3 Sí, 2 No) -> Gini = 0.4800. RAMA IMPURA, requiere partición.")

print("\n" + "=" * 70)
print("4. PARTICIÓN DEL SUBNODO Urgencia == 'Media' (5 correos)")
print("=" * 70)
df_media = df[df['Urgencia'] == 'Media']
sub_vars = ['Remitente', 'Adjunto', 'Enlace_corto']
sub_resultados = {}

for var in sub_vars:
    print(f"\n>>> Sub-evaluación con: {var}")
    gini_pond = 0.0
    for val, grupo in df_media.groupby(var):
        g = calcular_gini(grupo['Phishing'])
        cnt_si = (grupo['Phishing'] == 'Sí').sum()
        cnt_no = (grupo['Phishing'] == 'No').sum()
        peso = len(grupo) / len(df_media)
        gini_pond += peso * g
        print(f"   Sub-rama '{val}': {len(grupo)} correos (Sí={cnt_si}, No={cnt_no}) -> Gini = {g:.4f}")
    sub_resultados[var] = gini_pond
    print(f"   => Gini ponderado ({var}) en rama Media = {gini_pond:.4f}")

sub_raiz = min(sub_resultados, key=sub_resultados.get)
print(f"\nEn la rama Media, la variable con menor Gini ponderado es '{sub_raiz}' (Gini = {sub_resultados[sub_raiz]:.4f}).")
print("Como el Gini es 0.0000, ¡esta partición separa perfectamente las clases!")

print("\n" + "=" * 70)
print("5. ESTRUCTURA COMPLETA DEL ÁRBOL MANUAL")
print("=" * 70)
arbol_ascii = """
                [ ¿Urgencia? ] (Gini=0.500)
               /       |      \\
        Alta  /  Media |       \\ Baja
             /         |        \\
      [ Phishing = Sí ]|        [ Phishing = No ]
      (4/4 Sí, Gini=0) |        (5/5 No, Gini=0)
                       |
               [ ¿Remitente? ] (Gini=0.480)
              /               \\
  Desconocido/                 \\ Conocido
            /                   \\
   [ Phishing = Sí ]        [ Phishing = No ]
   (3/3 Sí, Gini=0)         (2/2 No, Gini=0)
"""
print(arbol_ascii)

print("=" * 70)
print("6. REGLAS LÓGICAS 'SI... ENTONCES' DEL ÁRBOL A MANO")
print("=" * 70)
reglas = """
REGLA 1:
  SI Urgencia = "Alta" 
  ENTONCES Phishing = "Sí"

REGLA 2:
  SI Urgencia = "Baja" 
  ENTONCES Phishing = "No"

REGLA 3:
  SI Urgencia = "Media" Y Remitente = "Desconocido" 
  ENTONCES Phishing = "Sí"

REGLA 4:
  SI Urgencia = "Media" Y Remitente = "Conocido" 
  ENTONCES Phishing = "No"

REGLA POR DEFECTO (EN OTRO CASO):
  EN OTRO CASO 
  ENTONCES Phishing = "No" (se asigna la clase más conservadora / no maliciosa)
"""
print(reglas)
print("=" * 70)
