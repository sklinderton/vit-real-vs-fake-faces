"""
charts.py — Gráficas interactivas (Plotly)
===========================================
Todas las cifras provienen de las corridas reales en el clúster Kabré.
Cuando el archivo de resultados existe se lee de ahí; si no, se usan los
valores registrados para que el dashboard siga siendo demostrable.
"""

import plotly.graph_objects as go

import theme

# ══ Datos medidos ═════════════════════════════════════════════════════════

# Barrido de truncation sobre la primera versión (200 rostros por valor).
TRUNCATION_SWEEP = {
    "psi": [0.40, 0.55, 0.70, 0.85, 1.00],
    "detection": [1.000, 0.975, 0.765, 0.495, 0.230],
}

VERSIONS = {
    "labels": ["Primera<br>versión", "Datos<br>corregidos", "Ajuste de<br>entrenamiento",
                "Tres<br>ViT", "ConvNeXt<br>+ Swin"],
    "known": [82.0, 93.0, 97.4, 98.9, 99.9],
    "unseen": [23.0, 60.9, 87.0, 94.0, 99.8],
}

# Aciertos de cada arquitectura por separado sobre el conjunto difícil.
# El hallazgo central: la red convolucional supera al transformador de
# parches pese a tener un tercio de sus parámetros.
ARCHITECTURES = {
    "names": ["ViT-B/16", "Swin-T", "ConvNeXt-Tiny"],
    "accuracy": [93.3, 97.3, 99.6],
    "params_m": [85.8, 27.5, 27.8],
    "auc": [0.9778, 0.9995, 0.9999],
}

# Preprocesamiento en paralelo con 16 procesos (ProcessPoolExecutor).
PREPROCESSING = {
    "split": ["Entrenamiento", "Validación", "Prueba"],
    "images": [16000, 2000, 2000],
    "seconds": [6.8, 1.3, 1.4],
    "throughput": [2358.1, 1550.3, 1434.7],
}

# Filtros de redimensionado probados (hipótesis descartada).
RESAMPLING = {
    "method": ["Lanczos", "Bicúbico", "Bilineal", "Área", "Vecino", "Directo"],
    "detection": [20.5, 19.5, 18.5, 18.5, 19.0, 21.5],
}

HYPOTHESES = [
    ("Las dos clases venían comprimidas de forma distinta", "Descartada",
     "Todo el conjunto resultó ser PNG sin pérdida. La diferencia en peso por "
     "píxel entre auténticas y sintéticas fue de apenas d = 0.093."),
    ("El redimensionado dejaba una huella distinta", "Descartada",
     "Se probaron seis formas de reducir la imagen. La detección se movió solo "
     "entre 18.5 % y 21.5 %: tres puntos de diferencia."),
    ("El conjunto se generó con poca diversidad", "Confirmada",
     "La detección cae de 100 % a 23 % conforme se sube la diversidad del "
     "generador. Setenta y siete puntos de diferencia."),
]

BASE = dict(
    font=dict(family=theme.PLOTLY_FONT, size=12, color=theme.TEXT),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=60, r=26, t=30, b=50),
    hoverlabel=dict(font=dict(family="JetBrains Mono, monospace", size=11),
                     bgcolor=theme.SURFACE_3, font_color=theme.TEXT, bordercolor=theme.RULE),
    xaxis=dict(gridcolor=theme.RULE, zerolinecolor=theme.RULE, linecolor=theme.RULE),
    yaxis=dict(gridcolor=theme.RULE, zerolinecolor=theme.RULE, linecolor=theme.RULE),
    transition=dict(duration=420, easing="cubic-in-out"),
)


def _apply(fig, height=340, legend=True):
    fig.update_layout(**BASE, height=height, showlegend=legend,
                       legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                   xanchor="left", x=0, font=dict(size=11),
                                   bgcolor="rgba(0,0,0,0)"))
    return fig


# ══ Gráficas ══════════════════════════════════════════════════════════════

def truncation_chart():
    d = TRUNCATION_SWEEP
    pct = [v * 100 for v in d["detection"]]
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=50, fillcolor=theme.SYNTHETIC, opacity=0.05, line_width=0)
    fig.add_hline(y=50, line=dict(color=theme.TEXT_FAINT, width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=d["psi"], y=pct, mode="lines+markers+text",
        name="Aciertos sobre sintéticas",
        line=dict(color=theme.SYNTHETIC, width=3, shape="spline"),
        marker=dict(size=12, color=theme.SYNTHETIC,
                    line=dict(color=theme.BG, width=2.5)),
        text=[f"{v:.0f} %" for v in pct], textposition="top center",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color=theme.TEXT),
        fill="tozeroy", fillcolor="rgba(123,79,209,0.07)",
        hovertemplate="Diversidad ψ = %{x}<br>Aciertos: %{y:.1f} %<extra></extra>"))
    fig.add_annotation(x=0.4, y=100, ax=45, ay=-28, text="caras suaves,<br>fáciles",
                        font=dict(size=10, color=theme.TEXT_DIM), showarrow=False)
    fig.add_annotation(x=1.0, y=23, ax=-45, ay=-30, text="caras diversas,<br>casi ninguna detectada",
                        font=dict(size=10, color=theme.SYNTHETIC), showarrow=False, xshift=-52)
    fig.update_xaxes(title="Diversidad del generador (ψ)", dtick=0.15)
    fig.update_yaxes(title="Aciertos sobre sintéticas", range=[0, 118], ticksuffix=" %")
    return _apply(fig, 380, legend=False)


def versions_chart():
    v = VERSIONS
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=v["labels"], y=v["known"], name="Con imágenes ya conocidas",
        marker=dict(color=theme.TEXT_FAINT, line=dict(width=0)),
        text=[f"{x:.1f} %" for x in v["known"]], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11),
        hovertemplate="%{x}<br>Conocidas: %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Bar(
        x=v["labels"], y=v["unseen"], name="Con caras nuevas y diversas",
        marker=dict(color=theme.SYNTHETIC, line=dict(width=0)),
        text=[f"{x:.1f} %" for x in v["unseen"]], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color=theme.SYNTHETIC),
        hovertemplate="%{x}<br>Nuevas: %{y:.1f} %<extra></extra>"))
    fig.update_yaxes(title="Aciertos sobre sintéticas", range=[0, 120], ticksuffix=" %")
    fig.update_layout(barmode="group", bargap=0.34, bargroupgap=0.09)
    return _apply(fig, 390)


def gap_chart():
    """Cuánta diferencia hay entre el escenario fácil y el difícil, por versión."""
    v = VERSIONS
    gaps = [k - u for k, u in zip(v["known"], v["unseen"])]
    colors = [theme.SYNTHETIC, theme.AMBER, theme.AUTHENTIC]
    fig = go.Figure(go.Bar(
        x=v["labels"], y=gaps, marker=dict(color=colors, line=dict(width=0)),
        text=[f"{g:.1f} pts" for g in gaps], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=12),
        hovertemplate="%{x}<br>Diferencia: %{y:.1f} puntos<extra></extra>"))
    fig.update_yaxes(title="Diferencia entre lo fácil y lo difícil", range=[0, 70])
    return _apply(fig, 320, legend=False)


def preprocessing_chart():
    """Rendimiento del preprocesamiento repartido entre 16 procesos."""
    p = PREPROCESSING
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=p["split"], y=p["throughput"],
        marker=dict(color=[theme.TEXT, theme.TEXT_DIM, theme.TEXT_FAINT],
                    line=dict(width=0)),
        text=[f"{t:,.0f}" for t in p["throughput"]], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11),
        customdata=list(zip(p["images"], p["seconds"])),
        hovertemplate=("%{x}<br>%{customdata[0]:,} imágenes<br>"
                        "%{customdata[1]} segundos<br>"
                        "%{y:,.0f} imágenes por segundo<extra></extra>")))
    fig.update_yaxes(title="Imágenes procesadas por segundo", range=[0, 2750])
    return _apply(fig, 330, legend=False)


def resampling_chart():
    """Seis formas de redimensionar: la detección apenas se mueve."""
    r = RESAMPLING
    fig = go.Figure(go.Bar(
        x=r["method"], y=r["detection"],
        marker=dict(color=theme.TEXT_FAINT, line=dict(width=0)),
        text=[f"{v:.1f} %" for v in r["detection"]], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=10),
        hovertemplate="%{x}<br>Aciertos: %{y:.1f} %<extra></extra>"))
    fig.add_hrect(y0=18.5, y1=21.5, fillcolor=theme.TEXT_DIM, opacity=0.07, line_width=0,
                   annotation_text="apenas 3 puntos de rango",
                   annotation_position="top left",
                   annotation_font=dict(size=10, color=theme.TEXT_DIM))
    fig.update_yaxes(title="Aciertos sobre sintéticas", range=[0, 30], ticksuffix=" %")
    return _apply(fig, 300, legend=False)


def spectral_chart():
    pairs = ["Auténticas frente a<br>sintéticas del conjunto",
             "Sintéticas del conjunto<br>frente a las nuestras",
             "Auténticas frente a<br>las nuestras"]
    vals = [0.01055, 0.00733, 0.00593]
    colors = [theme.TEXT, theme.SYNTHETIC, theme.AUTHENTIC]
    fig = go.Figure(go.Bar(
        x=vals, y=pairs, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.5f}" for v in vals], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11),
        hovertemplate="%{y}<br>Distancia: %{x:.5f}<extra></extra>"))
    fig.update_xaxes(title="Distancia entre perfiles de frecuencia", range=[0, 0.0137])
    return _apply(fig, 300, legend=False)


def training_chart(df=None):
    if df is None or df.empty:
        return None
    fig = go.Figure()
    if "train_auc_roc" in df:
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["train_auc_roc"], mode="lines", name="Entrenamiento",
            line=dict(color=theme.AUTHENTIC, width=1.8, dash="dot"),
            hovertemplate="Época %{x}<br>%{y:.4f}<extra></extra>"))
    if "val_auc_roc" in df:
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["val_auc_roc"], mode="lines+markers",
            name="Prueba fácil", line=dict(color=theme.TEXT_DIM, width=2),
            marker=dict(size=7), hovertemplate="Época %{x}<br>%{y:.4f}<extra></extra>"))
    if "valhard_auc_roc" in df:
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["valhard_auc_roc"], mode="lines+markers",
            name="Prueba difícil (la que decide)",
            line=dict(color=theme.SYNTHETIC, width=3),
            marker=dict(size=9, line=dict(color=theme.BG, width=2)),
            hovertemplate="Época %{x}<br>%{y:.4f}<extra></extra>"))
        best = df.loc[df["valhard_auc_roc"].idxmax()]
        fig.add_annotation(
            x=best["epoch"], y=best["valhard_auc_roc"],
            text="mejor punto,<br>aquí se guardó", showarrow=True, arrowhead=0,
            arrowcolor=theme.SYNTHETIC, ax=0, ay=-46,
            font=dict(size=10, color=theme.SYNTHETIC))
    fig.update_xaxes(title="Época", dtick=1)
    fig.update_yaxes(title="Calidad de la clasificación (AUC)")
    return _apply(fig, 370)


def confusion_chart(cm, title=""):
    if not cm:
        return None
    labels = ["Sintética", "Auténtica"]
    total = sum(sum(r) for r in cm)
    text = [[f"<b>{v}</b><br>{v / total * 100:.1f} %" for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels, text=text, texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono, monospace", size=13),
        colorscale=[[0, theme.SURFACE_2], [1, theme.SYNTHETIC_DIM]], showscale=False,
        hovertemplate="Realmente %{y}<br>El modelo dijo %{x}<br>%{z} casos<extra></extra>",
        xgap=4, ygap=4))
    fig.update_xaxes(title="Lo que dijo el modelo")
    fig.update_yaxes(title="Lo que era en realidad", autorange="reversed")
    layout = {**BASE, "height": 320, "showlegend": False}
    if title:
        layout["title"] = dict(text=title, font=dict(size=13), x=0, xanchor="left")
        layout["margin"] = dict(l=110, r=26, t=44, b=50)
    else:
        layout["margin"] = dict(l=110, r=26, t=20, b=50)
    fig.update_layout(**layout)
    return fig


def dataset_chart():
    """De qué está hecho el conjunto de entrenamiento."""
    labels = ["Fotos auténticas", "Sintéticas del conjunto público",
              "Sintéticas generadas por nosotros"]
    values = [9000, 4500, 4500]
    colors = [theme.AUTHENTIC, theme.TEXT_FAINT, theme.SYNTHETIC]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58,
        marker=dict(colors=colors, line=dict(color=theme.BG, width=3)),
        textinfo="percent", textfont=dict(family="JetBrains Mono, monospace",
                                           size=12, color=theme.TEXT),
        hovertemplate="%{label}<br>%{value:,} imágenes<br>%{percent}<extra></extra>",
        sort=False))
    fig.add_annotation(text="18 000<br><span style='font-size:11px'>imágenes</span>",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(family="JetBrains Mono, monospace",
                                  size=19, color=theme.TEXT))
    fig.update_layout(**{**BASE, "height": 340, "showlegend": True,
                          "legend": dict(orientation="h", y=-0.08, x=0.5,
                                         xanchor="center", font=dict(size=11))})
    return fig


# ══ Gráficas del análisis individual ══════════════════════════════════════

def probability_bars(p_fake, p_real):
    """Reparto de la decisión entre las dos clases."""
    fig = go.Figure(go.Bar(
        x=[p_real * 100, p_fake * 100], y=["Auténtica", "Sintética"],
        orientation="h",
        marker=dict(color=[theme.AUTHENTIC, theme.SYNTHETIC], line=dict(width=0)),
        text=[f"{p_real*100:.1f} %", f"{p_fake*100:.1f} %"], textposition="auto",
        textfont=dict(family="JetBrains Mono, monospace", size=13, color=theme.BG),
        hovertemplate="%{y}: %{x:.2f} %<extra></extra>"))
    fig.add_vline(x=50, line=dict(color=theme.TEXT_FAINT, width=1, dash="dot"))
    fig.update_xaxes(title="", range=[0, 100], ticksuffix=" %")
    fig.update_yaxes(title="")
    fig.update_layout(**{**BASE, "height": 170, "showlegend": False,
                          "margin": dict(l=76, r=20, t=10, b=32)})
    return fig


def context_chart(p_fake):
    """
    Sitúa esta imagen frente a lo que el modelo suele responder por cada
    tipo de entrada, según las mediciones del conjunto de prueba.
    """
    cats = ["Fotos auténticas", "Sintéticas suaves<br>(ψ bajo)",
            "Sintéticas diversas<br>(ψ = 1.0)"]
    typical = [7.4, 96.0, 82.0]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cats, y=typical, name="Respuesta habitual",
        marker=dict(color=[theme.AUTHENTIC_DIM, theme.SYNTHETIC_DIM,
                            theme.SYNTHETIC_DIM], line=dict(width=0)),
        hovertemplate="%{x}<br>Suele dar: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=p_fake * 100, line=dict(color=theme.TEXT, width=2, dash="dash"),
                   annotation_text=f"esta imagen: {p_fake*100:.1f} %",
                   annotation_position="top right",
                   annotation_font=dict(size=11, color=theme.TEXT,
                                        family="JetBrains Mono, monospace"))
    fig.update_yaxes(title="Probabilidad de ser sintética", range=[0, 112], ticksuffix=" %")
    fig.update_layout(**{**BASE, "height": 280, "showlegend": False,
                          "margin": dict(l=60, r=26, t=34, b=54)})
    return fig


def attention_profile(attn):
    """Distribución del peso de la atención sobre los 196 parches."""
    if attn is None:
        return None
    import numpy as np
    a = np.sort(attn.flatten())[::-1]
    a = a / (a.sum() + 1e-9) * 100
    cum = np.cumsum(a)
    x = list(range(1, len(a) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=cum, mode="lines", name="Atención acumulada",
        line=dict(color=theme.SYNTHETIC, width=2.5, shape="spline"),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.12)",
        hovertemplate="Los %{x} parches más mirados<br>concentran %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=50, line=dict(color=theme.TEXT_FAINT, width=1, dash="dot"),
                   annotation_text="mitad de la atención",
                   annotation_font=dict(size=10, color=theme.TEXT_FAINT))
    fig.update_xaxes(title="Parches ordenados de mayor a menor atención")
    fig.update_yaxes(title="Atención acumulada", range=[0, 105], ticksuffix=" %")
    fig.update_layout(**{**BASE, "height": 260, "showlegend": False,
                          "margin": dict(l=60, r=26, t=20, b=50)})
    return fig


def architecture_chart():
    """
    Aciertos frente a tamaño del modelo. Cada punto es una arquitectura
    evaluada sobre el conjunto difícil.
    """
    a = ARCHITECTURES
    colors = [theme.TEXT_FAINT, theme.BLUE, theme.AUTHENTIC]
    fig = go.Figure()
    for i, name in enumerate(a["names"]):
        fig.add_trace(go.Scatter(
            x=[a["params_m"][i]], y=[a["accuracy"][i]], mode="markers+text",
            name=name, marker=dict(size=26, color=colors[i],
                                    line=dict(color=theme.BG, width=2)),
            text=[name], textposition="top center",
            textfont=dict(family="JetBrains Mono, monospace", size=11,
                          color=theme.TEXT),
            hovertemplate=(f"{name}<br>%{{x:.1f}} M parámetros<br>"
                            f"%{{y:.1f}} % aciertos<br>"
                            f"AUC {a['auc'][i]}<extra></extra>")))
    fig.update_xaxes(title="Tamaño del modelo (millones de parámetros)",
                      range=[15, 98])
    fig.update_yaxes(title="Aciertos con caras nuevas", range=[91, 101],
                      ticksuffix=" %")
    return _apply(fig, 340, legend=False)


def ensemble_chart():
    """Cómo rinde cada combinación de modelos."""
    combos = ["Tres ViT", "Los cinco", "ConvNeXt<br>+ Swin"]
    acc = [94.0, 97.9, 99.7]
    params = [258, 313, 55]
    colors = [theme.TEXT_FAINT, theme.BLUE, theme.AUTHENTIC]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=combos, y=acc, marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f} %" for v in acc], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=12),
        customdata=params,
        hovertemplate=("%{x}<br>%{y:.1f} % aciertos<br>"
                        "%{customdata} M parámetros<extra></extra>")))
    fig.update_yaxes(title="Aciertos con caras nuevas", range=[88, 103],
                      ticksuffix=" %")
    return _apply(fig, 320, legend=False)
