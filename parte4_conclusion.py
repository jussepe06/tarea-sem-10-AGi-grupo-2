"""
================================================================================
PARTE 4: CONCLUSIÓN Y RECOMENDACIÓN PRÁCTICA
================================================================================
Curso: Agentes Inteligentes - Semana 10 (USIL)
Objetivo:
  Redactar un párrafo de 4 o 5 líneas, escrito para alguien que no sabe de IA
  (un tutor universitario / director de carrera), respondiendo:
    - ¿Qué modelo recomiendas?
    - ¿Qué variables debe vigilar?
    - ¿Qué acción concreta propones?
================================================================================
"""

conclusion_texto = """
RECOMENDACIÓN PARA EL EQUIPO DE TUTORÍA UNIVERSITARIA (NO ESPECIALISTAS EN IA):

"Para la prevención oportuna del abandono estudiantil, recomendamos implementar el 
modelo de Random Forest (bosque de decisión), ya que demostró mayor capacidad para 
detectar a los estudiantes en riesgo real (alcanzando un F1-score del 76.2% y reduciendo 
los casos inadvertidos a solo 6). Los tutores deben vigilar prioritariamente el promedio 
ponderado acumulado (especialmente si desciende de 10) y la cantidad de cursos reprobados 
(2 o más materias). Como acción concreta, proponemos activar un sistema de alerta temprana 
en la semana 4 de cada ciclo que agende citas de mentoría académica personalizada y facilidades 
de reprogramación de pagos antes de que la deserción sea irreversible."
"""

print("=" * 75)
print("PARTE 4: CONCLUSIÓN EJECUTIVA (4-5 LÍNEAS PARA NO ESPECIALISTAS)")
print("=" * 75)
print(conclusion_texto)
print("=" * 75)
