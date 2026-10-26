"""
Generador del diagrama visual del Árbol a Mano de la Parte 1
Guarda la imagen 'arbol_a_mano_parte1.png' para incluirla en el cuaderno (.ipynb).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Título
ax.text(50, 95, "ÁRBOL DE DECISIÓN A MANO — PARTE 1 (ÍNDICE GINI)", 
        fontsize=15, weight='bold', ha='center', color='#002060')
ax.text(50, 91, "Clasificación de 14 Correos de Phishing | Criterio: Mínimo Gini Ponderado", 
        fontsize=11, ha='center', color='#555555')

def draw_box(x, y, w, h, title, text, bg_color, border_color='#002060'):
    rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                  boxstyle="round,pad=0.8",
                                  ec=border_color, fc=bg_color, lw=2)
    ax.add_patch(rect)
    ax.text(x, y + 2.2, title, fontsize=10.5, weight='bold', ha='center', color=border_color)
    ax.text(x, y - 2.2, text, fontsize=9, ha='center', color='#222222')

def draw_arrow(x1, y1, x2, y2, label="", color='#002060'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=8, shrink=0.05))
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + 2.5
        ax.text(mid_x, mid_y, label, fontsize=9.5, weight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#c7d5e4", lw=1))

# 1. Nodo Raíz
draw_box(50, 78, 28, 11, "¿Urgencia?", "Total = 14 | Gini = 0.5000\n(7 Sí, 7 No)", "#DCE6F2", "#002060")

# 2. Ramas de la raíz
# Rama Alta (Izquierda) -> Hoja Phishing = Sí
draw_arrow(40, 73, 20, 56, "Alta (4/14)")
draw_box(20, 50, 24, 11, "Phishing = SÍ", "Total = 4 | Gini = 0.0000\n(4 Sí, 0 No) — ¡HOJA PURA!", "#dcfce7", "#15803d")

# Rama Baja (Derecha) -> Hoja Phishing = No
draw_arrow(60, 73, 80, 56, "Baja (5/14)")
draw_box(80, 50, 24, 11, "Phishing = NO", "Total = 5 | Gini = 0.0000\n(0 Sí, 5 No) — ¡HOJA PURA!", "#dcfce7", "#15803d")

# Rama Media (Centro) -> Nodo Interno ¿Remitente?
draw_arrow(50, 72, 50, 56, "Media (5/14)")
draw_box(50, 50, 26, 11, "¿Remitente?", "Total = 5 | Gini = 0.4800\n(3 Sí, 2 No) — Impuro", "#fef3c7", "#b45309")

# 3. Sub-ramas de Remitente
# Desconocido -> Phishing = Sí
draw_arrow(44, 44, 35, 26, "Desconocido (3/5)")
draw_box(35, 20, 24, 11, "Phishing = SÍ", "Total = 3 | Gini = 0.0000\n(3 Sí, 0 No) — ¡HOJA PURA!", "#dcfce7", "#15803d")

# Conocido -> Phishing = No
draw_arrow(56, 44, 65, 26, "Conocido (2/5)")
draw_box(65, 20, 24, 11, "Phishing = NO", "Total = 2 | Gini = 0.0000\n(0 Sí, 2 No) — ¡HOJA PURA!", "#dcfce7", "#15803d")

# Cuadro explicativo inferior
info_text = (
    "REGLAS LÓGICAS:\n"
    "1. SI Urgencia = 'Alta' ENTONCES Phishing = 'Sí'\n"
    "2. SI Urgencia = 'Baja' ENTONCES Phishing = 'No'\n"
    "3. SI Urgencia = 'Media' Y Remitente = 'Desconocido' ENTONCES Phishing = 'Sí'\n"
    "4. SI Urgencia = 'Media' Y Remitente = 'Conocido' ENTONCES Phishing = 'No'\n"
    "5. EN OTRO CASO ENTONCES Phishing = 'No' (Regla preventiva)"
)
ax.text(50, 5, info_text, fontsize=8.8, family='monospace', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#002060", lw=1.5))

plt.tight_layout()
plt.savefig("arbol_a_mano_parte1.png", dpi=300)
print("¡arbol_a_mano_parte1.png generado exitosamente!")
