"""
Generador de BD_Semana10_Riesgos.xlsx
Genera los 5 conjuntos de datos sintéticos (200 filas c/u) con 5 variables numéricas y 1 objetivo binario,
más la hoja 'Phishing_mano' (14 filas) y las hojas de documentación 'Diccionario' y 'Fuentes'.
Distribución balanceada (~35-40% clase positiva) para asegurar métricas didácticas y realistas.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

# ------------------------------------------------------------------------------
# 1. Caso: Deserción Universitaria (Fuente: Minedu-SIRIES)
# ------------------------------------------------------------------------------
promedio = np.clip(np.random.normal(12.5, 2.5, n), 7.0, 19.5).round(2)
asistencia = np.clip(np.random.normal(78, 14, n), 35.0, 100.0).round(1)
cursos_rep = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.40, 0.25, 0.15, 0.12, 0.08])
retraso_pagos = np.random.choice([0, 1, 2, 3], size=n, p=[0.50, 0.25, 0.15, 0.10])
horas_estudio = np.clip(np.random.normal(12, 5, n), 2.0, 30.0).round(1)

score_desercion = (
    -0.45 * (promedio - 12.0) 
    - 0.05 * (asistencia - 75) 
    + 0.90 * cursos_rep 
    + 0.70 * retraso_pagos 
    - 0.12 * horas_estudio 
    - 0.40
)
prob_deserta = 1 / (1 + np.exp(-score_desercion))
deserta = (prob_deserta > 0.45).astype(int)

df_desercion = pd.DataFrame({
    'promedio_ponderado': promedio,
    'asistencia_pct': asistencia,
    'cursos_reprobados': cursos_rep,
    'retraso_pagos_meses': retraso_pagos,
    'horas_estudio_semanal': horas_estudio,
    'deserta': deserta
})

# ------------------------------------------------------------------------------
# 2. Caso: ETS (Fuente: INEI-ENDES 2020)
# ------------------------------------------------------------------------------
edad = np.random.randint(18, 50, size=n)
parejas = np.random.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.35, 0.25, 0.18, 0.11, 0.06, 0.05])
uso_condon = np.clip(np.random.normal(55, 28, n), 0, 100).round(0)
frecuencia_mes = np.random.randint(1, 25, size=n)
inicio_sexual = np.random.randint(14, 23, size=n)

score_ets = (
    0.50 * parejas 
    - 0.04 * uso_condon 
    + 0.05 * frecuencia_mes 
    - 0.10 * (inicio_sexual - 17) 
    - 0.50
)
prob_ets = 1 / (1 + np.exp(-score_ets))
its_positivo = (prob_ets > 0.45).astype(int)

df_ets = pd.DataFrame({
    'edad': edad,
    'parejas_sexuales': parejas,
    'uso_preservativo_pct': uso_condon,
    'relaciones_mes': frecuencia_mes,
    'edad_inicio_sexual': inicio_sexual,
    'its_positivo': its_positivo
})

# ------------------------------------------------------------------------------
# 3. Caso: Spam (Fuente: Statista / SpamAssassin)
# ------------------------------------------------------------------------------
longitud = np.random.randint(50, 2500, size=n)
palabras_urgencia = np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.30, 0.25, 0.20, 0.13, 0.08, 0.04])
enlaces = np.random.choice([0, 1, 2, 3, 4, 6], size=n, p=[0.25, 0.30, 0.22, 0.12, 0.07, 0.04])
mayusculas_pct = np.clip(np.random.normal(18, 16, n), 0, 90).round(1)
adjuntos = np.random.choice([0, 1, 2, 3], size=n, p=[0.50, 0.28, 0.14, 0.08])

score_spam = (
    0.70 * palabras_urgencia 
    + 0.55 * enlaces 
    + 0.05 * mayusculas_pct 
    + 0.45 * adjuntos 
    - 4.50
)
prob_spam = 1 / (1 + np.exp(-score_spam))
es_spam = (prob_spam > 0.45).astype(int)

df_spam = pd.DataFrame({
    'longitud_caracteres': longitud,
    'palabras_urgencia': palabras_urgencia,
    'enlaces_count': enlaces,
    'mayusculas_pct': mayusculas_pct,
    'adjuntos_count': adjuntos,
    'es_spam': es_spam
})

# ------------------------------------------------------------------------------
# 4. Caso: Phishing (Fuente: UCI Website Phishing)
# ------------------------------------------------------------------------------
longitud_url = np.random.randint(20, 160, size=n)
subdominios = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.30, 0.30, 0.22, 0.12, 0.06])
tiene_ip = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
tiene_arroba = np.random.choice([0, 1], size=n, p=[0.82, 0.18])
antiguedad_meses = np.random.randint(1, 48, size=n)

score_phishing = (
    0.035 * longitud_url 
    + 0.75 * subdominios 
    + 1.90 * tiene_ip 
    + 1.50 * tiene_arroba 
    - 0.08 * antiguedad_meses 
    - 4.60
)
prob_phishing = 1 / (1 + np.exp(-score_phishing))
es_phishing = (prob_phishing > 0.45).astype(int)

df_phishing = pd.DataFrame({
    'longitud_url': longitud_url,
    'subdominios_count': subdominios,
    'tiene_ip': tiene_ip,
    'tiene_arroba': tiene_arroba,
    'antiguedad_dominio_meses': antiguedad_meses,
    'es_phishing': es_phishing
})

# ------------------------------------------------------------------------------
# 5. Caso: Extorsión (Fuente: Mininter / Observatorio de Seguridad)
# ------------------------------------------------------------------------------
ingreso_miles = np.clip(np.random.normal(14, 6.5, n), 2.5, 45.0).round(1)
incidencia_zona = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.12, 0.22, 0.32, 0.22, 0.12])
antiguedad_negocio = np.random.randint(1, 25, size=n)
camaras_seguridad = np.random.choice([0, 1], size=n, p=[0.50, 0.50])
afluencia_dia = np.random.randint(30, 450, size=n)

score_extorsion = (
    0.09 * ingreso_miles 
    + 0.80 * incidencia_zona 
    + 0.05 * antiguedad_negocio 
    - 1.10 * camaras_seguridad 
    + 0.004 * afluencia_dia 
    - 5.00
)
prob_extorsion = 1 / (1 + np.exp(-score_extorsion))
victima_extorsion = (prob_extorsion > 0.45).astype(int)

df_extorsion = pd.DataFrame({
    'ingreso_mensual_miles': ingreso_miles,
    'incidencia_zona_1_a_5': incidencia_zona,
    'antiguedad_negocio_anos': antiguedad_negocio,
    'camaras_seguridad': camaras_seguridad,
    'afluencia_clientes_dia': afluencia_dia,
    'victima_extorsion': victima_extorsion
})

# ------------------------------------------------------------------------------
# 6. Hoja Phishing_mano: 14 correos exactos de la Parte 1
# ------------------------------------------------------------------------------
phishing_mano_data = [
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
df_phishing_mano = pd.DataFrame(
    phishing_mano_data, 
    columns=['#', 'Urgencia', 'Remitente', 'Adjunto', 'Enlace corto', '¿Phishing?']
)

# ------------------------------------------------------------------------------
# 7. Hoja Diccionario
# ------------------------------------------------------------------------------
diccionario_data = [
    ['Desercion', 'promedio_ponderado', 'Promedio acumulado de notas del estudiante (escala 0-20)'],
    ['Desercion', 'asistencia_pct', 'Porcentaje global de asistencia a clases teóricas y prácticas (0-100%)'],
    ['Desercion', 'cursos_reprobados', 'Número de asignaturas reprobadas en ciclos previos'],
    ['Desercion', 'retraso_pagos_meses', 'Meses acumulados de mora en el pago de pensiones universitarias'],
    ['Desercion', 'horas_estudio_semanal', 'Horas semanales dedicadas a estudio autónomo fuera de aula'],
    ['Desercion', 'deserta', 'Objetivo: 1 = estudiante abandona estudios, 0 = continúa'],
    
    ['ETS', 'edad', 'Edad de la persona evaluada en años cumplidos'],
    ['ETS', 'parejas_sexuales', 'Número de parejas sexuales en los últimos 12 meses'],
    ['ETS', 'uso_preservativo_pct', 'Porcentaje estimado de relaciones con uso de preservativo'],
    ['ETS', 'relaciones_mes', 'Frecuencia promedio de relaciones sexuales por mes'],
    ['ETS', 'edad_inicio_sexual', 'Edad en años en la que inició su vida sexual'],
    ['ETS', 'its_positivo', 'Objetivo: 1 = resultado positivo a tamizaje, 0 = negativo'],
    
    ['Spam', 'longitud_caracteres', 'Longitud total del mensaje en caracteres'],
    ['Spam', 'palabras_urgencia', 'Frecuencia de términos de alarma/urgencia ("urgente", "ganaste", etc.)'],
    ['Spam', 'enlaces_count', 'Número de hipervínculos detectados en el correo'],
    ['Spam', 'mayusculas_pct', 'Porcentaje de caracteres escritos en mayúsculas sostenidas'],
    ['Spam', 'adjuntos_count', 'Cantidad de archivos adjuntos en el mensaje'],
    ['Spam', 'es_spam', 'Objetivo: 1 = mensaje clasificado como spam, 0 = correo legítimo'],
    
    ['Phishing', 'longitud_url', 'Cantidad de caracteres en la URL del enlace'],
    ['Phishing', 'subdominios_count', 'Número de subdominios presentes en el nombre de dominio'],
    ['Phishing', 'tiene_ip', '1 si el enlace usa dirección IP numérica directamente, 0 si usa dominio'],
    ['Phishing', 'tiene_arroba', '1 si la URL contiene el carácter "@" para enmascaramiento, 0 si no'],
    ['Phishing', 'antiguedad_dominio_meses', 'Meses transcurridos desde el registro del dominio en WHOIS'],
    ['Phishing', 'es_phishing', 'Objetivo: 1 = sitio malicioso que busca robo de credenciales, 0 = seguro'],
    
    ['Extorsion', 'ingreso_mensual_miles', 'Ingresos brutos mensuales del negocio en miles de soles (S/.)'],
    ['Extorsion', 'incidencia_zona_1_a_5', 'Nivel de criminalidad en la jurisdicción del negocio (1=bajo, 5=muy alto)'],
    ['Extorsion', 'antiguedad_negocio_anos', 'Años de funcionamiento ininterrumpido del establecimiento'],
    ['Extorsion', 'camaras_seguridad', '1 si cuenta con sistema de videovigilancia propio, 0 si no'],
    ['Extorsion', 'afluencia_clientes_dia', 'Promedio estimado de clientes atendidos por día'],
    ['Extorsion', 'victima_extorsion', 'Objetivo: 1 = negocio fue contactado/víctima de extorsión, 0 = no']
]
df_diccionario = pd.DataFrame(diccionario_data, columns=['Hoja', 'Columna', 'Descripcion'])

# ------------------------------------------------------------------------------
# 8. Hoja Fuentes
# ------------------------------------------------------------------------------
fuentes_data = [
    ['Desercion', 'Minedu - SIRIES', 'Sistema de Información de Respuesta a la Educación Superior, Perú. Factores de riesgo académico y financiero.'],
    ['ETS', 'INEI - ENDES 2020', 'Encuesta Demográfica y de Salud Familiar. Indicadores de salud reproductiva y prevención ITS.'],
    ['Spam', 'Statista / SpamAssassin Public Corpus', 'Estadísticas globales de ciberseguridad sobre volumen y morfología de correos no deseados.'],
    ['Phishing', 'UCI Machine Learning Repository - Website Phishing', 'Dataset público de atributos heurísticos y lexicales de URLs maliciosas.'],
    ['Extorsion', 'Observatorio de Criminalidad - Mininter Perú', 'Informes estadísticos sobre denuncias de extorsión a mypes y comercios urbanos.']
]
df_fuentes = pd.DataFrame(fuentes_data, columns=['Caso', 'Fuente_Oficial', 'Detalle'])

# Guardar en archivo Excel con formato profesional
with pd.ExcelWriter('BD_Semana10_Riesgos.xlsx', engine='openpyxl') as writer:
    df_desercion.to_excel(writer, sheet_name='Desercion', index=False)
    df_ets.to_excel(writer, sheet_name='ETS', index=False)
    df_spam.to_excel(writer, sheet_name='Spam', index=False)
    df_phishing.to_excel(writer, sheet_name='Phishing', index=False)
    df_extorsion.to_excel(writer, sheet_name='Extorsion', index=False)
    df_phishing_mano.to_excel(writer, sheet_name='Phishing_mano', index=False)
    df_diccionario.to_excel(writer, sheet_name='Diccionario', index=False)
    df_fuentes.to_excel(writer, sheet_name='Fuentes', index=False)

print("¡BD_Semana10_Riesgos.xlsx generado exitosamente con 8 hojas!")
for hoja, df_c in [('Desercion', df_desercion), ('ETS', df_ets), ('Spam', df_spam), ('Phishing', df_phishing), ('Extorsion', df_extorsion)]:
    target_col = df_c.columns[-1]
    pos = df_c[target_col].sum()
    print(f" - {hoja:10s}: {pos}/200 ({pos/200:.1%}) casos positivos")
