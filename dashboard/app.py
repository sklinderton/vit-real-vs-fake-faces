"""
app.py — ¿Esta cara existe?
============================
Dashboard del proyecto de Computación Paralela y Distribuida, Universidad LEAD.

Clasifica rostros como fotografía auténtica o imagen generada por StyleGAN3,
con un Vision Transformer ajustado sobre el clúster Kabré del CeNAT.

Ejecutar:
    streamlit run app.py
"""

import json
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

import charts
import engine
import theme

APP_DIR = Path(__file__).parent
ASSETS = APP_DIR / "assets"
METRICS_FILE = ASSETS / "evaluation_summary_ensemble.json"
TRAINING_LOG = ASSETS / "training_log_cnx.csv"
POOL_DIR = ASSETS / "stylegan3_pool"

st.set_page_config(page_title="¿Esta cara existe? · LEAD",
                    page_icon="◑", layout="wide",
                    initial_sidebar_state="collapsed")


# ══ Recursos ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Cargando el ensamble…")
def get_model():
    return engine.load_ensemble(ASSETS)


@st.cache_data
def get_metrics():
    return json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else None


@st.cache_data
def get_training_log():
    return pd.read_csv(TRAINING_LOG) if TRAINING_LOG.exists() else None


@st.cache_data
def get_pool():
    m = POOL_DIR / "manifest.json"
    return json.loads(m.read_text()) if m.exists() else None


# ══ Componentes ═══════════════════════════════════════════════════════════

def log_entry(res, source):
    """Registra un análisis en el historial de la sesión."""
    st.session_state.log.insert(0, {
        "Hora": datetime.now().strftime("%H:%M:%S"),
        "Origen": source,
        "Veredicto": "Sintética" if res["p_fake"] > 0.5 else "Auténtica",
        "P(sintética)": round(res["p_fake"], 4),
    })


def hero(meta, metrics):
    st.markdown(
        '<div class="hero rise">'
        '<div class="eyebrow">Universidad LEAD · Computación Paralela y Distribuida</div>'
        '<h1>¿Esta cara <span class="accent">existe</span>?</h1>'
        '<p class="lede">Hoy cualquiera puede fabricar el retrato de una persona '
        'que nunca nació. Este sistema acierta en 997 de cada 1 000 — pero la '
        'primera versión solo detectaba 23 de cada 100, y averiguar por qué '
        'fue el verdadero trabajo.</p>'
        '</div>', unsafe_allow_html=True)

    hard = ((metrics or {}).get("evaluations", {})
            .get("test_hard", {}).get("tuned", {}))
    params = f'{meta["params"] / 1e6:.0f} M' if meta else "—"
    arch = " + ".join(meta["names"]) if meta else "—"
    cells = [
        ("Modelos", arch, False),
        ("Parámetros", params, False),
        ("Entrenado en", "NVIDIA L40S", False),
        ("Aciertos", f'{hard.get("accuracy", 0) * 100:.1f} %' if hard else "99.7 %", True),
        ("Detecta sintéticas", f'{hard.get("recall_fake", 0) * 100:.1f} %' if hard else "99.8 %", True),
    ]
    html = '<div class="datastrip rise rise-2">'
    for k, v, hl in cells:
        html += (f'<div class="cell"><span class="k">{k}</span>'
                 f'<span class="v{" hl" if hl else ""}">{v}</span></div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(
        '<div class="caption-mono">Medido contra rostros creados con semillas que el '
        'modelo nunca vio, a máxima diversidad del generador — el peor escenario posible.</div>',
        unsafe_allow_html=True)


def render_result(res, provenance=None, source="—", key="main", meta_names=None):
    """
    Dibuja el panel de resultado. Recibe el resultado ya calculado desde
    session_state, de modo que interactuar con el toggle de atención
    no borra el análisis: Streamlit re-ejecuta el script en cada
    interacción, y si el resultado dependiera de un botón, el botón
    valdría False en la nueva pasada y el panel desaparecería.
    """
    p_fake = res["p_fake"]
    thr = engine.DECISION_THRESHOLD
    is_fake = p_fake > thr
    verdict = "Sintética" if is_fake else "Auténtica"
    color = theme.SYNTHETIC if is_fake else theme.AUTHENTIC
    conf = max(p_fake, res["p_real"])
    borderline = abs(p_fake - thr) < 0.15

    st.markdown("---")
    left, right = st.columns([1, 1.08], gap="large")

    with left:
        show_attn = st.toggle("Resaltar dónde miró el modelo", value=False,
                               key=f"attn_toggle_{key}")
        img = res["image"]
        if show_attn and res.get("attention") is not None:
            overlay = engine.attention_overlay(img, res["attention"])
            st.image(overlay, use_container_width=True)
            st.markdown('<div class="caption-mono">Las zonas claras son las que más '
                         'pesaron en la decisión.</div>', unsafe_allow_html=True)
        else:
            st.image(img, use_container_width=True)
            st.markdown('<div class="caption-mono">Imagen ajustada a 224 × 224 píxeles, '
                         'el tamaño que espera el modelo.</div>', unsafe_allow_html=True)

    with right:
        st.markdown(f'<div class="card-title">Veredicto</div>'
                     f'<div class="verdict-word" style="color:{color}">{verdict}</div>',
                     unsafe_allow_html=True)
        st.markdown(engine.verdict_dial(p_fake), unsafe_allow_html=True)
        st.markdown(
            f'<div class="readout-grid">'
            f'<span class="k">Probabilidad de ser sintética</span><span class="v">{p_fake:.4f}</span>'
            f'<span class="k">Probabilidad de ser auténtica</span><span class="v">{res["p_real"]:.4f}</span>'
            f'<span class="k">Seguridad del modelo</span><span class="v">{conf:.1%}</span>'
            f'<span class="k">Punto de corte</span><span class="v">{thr:.3f}</span>'
            f'</div>', unsafe_allow_html=True)

        if res.get("per_model") and len(res["per_model"]) > 1 and meta_names:
            partes = " · ".join(
                f"{n.split('-')[0]} {p:.3f}"
                for n, p in zip(meta_names, res["per_model"]))
            st.markdown(f'<div class="caption-mono">Cada modelo por separado: '
                         f'{partes}</div>', unsafe_allow_html=True)
        if provenance:
            st.markdown(f'<div class="provenance">{provenance}</div>', unsafe_allow_html=True)
        if borderline:
            st.markdown('<div class="warnbox"><b>Resultado dudoso.</b> La probabilidad '
                         'está cerca del punto de corte, así que el modelo no tiene una '
                         'opinión clara. Conviene no tomarlo como conclusión.</div>',
                         unsafe_allow_html=True)

    # ── Gráficas del análisis ────────────────────────────────────────────
    st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
    g1, g2 = st.columns([1, 1], gap="large")
    with g1:
        st.markdown('<div class="card-title">Reparto de la decisión</div>',
                     unsafe_allow_html=True)
        st.plotly_chart(charts.probability_bars(p_fake, res["p_real"]),
                         use_container_width=True, config={"displayModeBar": False})
        stats = engine.attention_stats(res.get("attention"))
        if stats:
            st.markdown(
                f'<div class="caption-mono">Concentración de la mirada: los parches '
                f'más atendidos (el 10 % superior) acaparan '
                f'<b style="color:{theme.TEXT}">{stats["top10_share"]*100:.1f} %</b> '
                f'del peso total.</div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="card-title">Comparado con lo habitual</div>',
                     unsafe_allow_html=True)
        st.plotly_chart(charts.context_chart(p_fake), use_container_width=True,
                         config={"displayModeBar": False})

    prof = charts.attention_profile(res.get("attention"))
    if prof is not None:
        with st.expander("Ver cómo repartió la atención"):
            st.plotly_chart(prof, use_container_width=True,
                             config={"displayModeBar": False})
            st.markdown('<div class="caption-mono">Si la curva sube muy rápido, el '
                         'modelo se fijó en unas pocas zonas concretas. Si sube despacio, '
                         'repartió la mirada por toda la cara.</div>',
                         unsafe_allow_html=True)


# ══ Secciones ═════════════════════════════════════════════════════════════

def section_analyze(models, device, pool):
    st.markdown("#### Pon a prueba el modelo")
    st.markdown('<div class="note">Elige de dónde sacar la imagen. La cámara se '
                 'enciende solo si escoges esa opción.</div>', unsafe_allow_html=True)

    source = st.radio("Fuente", ["Subir archivo", "Tomar foto", "Generar una cara"],
                       horizontal=True, label_visibility="collapsed", key="src")

    # El resultado vive en session_state para que las interacciones
    # posteriores (toggle de atención, expander) no lo borren.
    slot = f"result_{source}"

    if source == "Subir archivo":
        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
        up = st.file_uploader("Imagen", type=["jpg", "jpeg", "png"],
                               label_visibility="collapsed")
        if up is not None:
            sig = f"{up.name}:{up.size}"
            if st.session_state.get(f"{slot}_sig") != sig:
                with st.spinner("Analizando…"):
                    res = engine.analyze(Image.open(up), models, device)
                st.session_state[slot] = res
                st.session_state[f"{slot}_sig"] = sig
                st.session_state[f"{slot}_prov"] = f"Archivo cargado · {up.name}"
                log_entry(res, "Archivo")
        if st.session_state.get(slot) is not None:
            render_result(st.session_state[slot],
                           st.session_state.get(f"{slot}_prov"), "Archivo", "file",
                           st.session_state.get("meta_names"))

    elif source == "Tomar foto":
        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
        shot = st.camera_input("Captura", label_visibility="collapsed")
        if shot is not None:
            sig = str(shot.size)
            if st.session_state.get(f"{slot}_sig") != sig:
                with st.spinner("Analizando…"):
                    res = engine.analyze(Image.open(shot), models, device)
                st.session_state[slot] = res
                st.session_state[f"{slot}_sig"] = sig
                st.session_state[f"{slot}_prov"] = "Capturado con la cámara de este dispositivo"
                log_entry(res, "Cámara")
        if st.session_state.get(slot) is not None:
            render_result(st.session_state[slot],
                           st.session_state.get(f"{slot}_prov"), "Cámara", "cam",
                           st.session_state.get("meta_names"))

    else:
        if pool is None:
            st.warning("Faltan los rostros generados. Copia `assets/stylegan3_pool/` "
                        "desde el clúster.")
            return
        st.markdown(
            f'<div class="note">Hay {pool["n_images"]} caras creadas por StyleGAN3 en '
            f'el clúster, con semillas que el modelo nunca vio. El control de diversidad '
            f'cambia cuánto se aleja cada cara del promedio: en el mínimo salen rostros '
            f'suaves y parecidos entre sí; en el máximo, caras con textura y carácter, '
            f'mucho más difíciles de descubrir.</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1.5, 1])
        with c1:
            opts = ["Al azar"] + [f"ψ = {p}" for p in pool["psi_values"]]
            choice = st.selectbox("Diversidad de la cara", opts)
        with c2:
            st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)
            draw = st.button("Crear una cara nueva", use_container_width=True)

        if draw:
            cand = pool["images"]
            if choice != "Al azar":
                t = float(choice.split("=")[1])
                cand = [e for e in cand if abs(e["psi"] - t) < 1e-6]
            entry = random.choice(cand)
            with st.spinner("Analizando…"):
                res = engine.analyze(Image.open(POOL_DIR / entry["filename"]),
                                      models, device)
            st.session_state[slot] = res
            st.session_state[f"{slot}_prov"] = (
                f"StyleGAN3 · semilla {entry['seed']} · diversidad ψ = {entry['psi']}<br>"
                f"Esta persona no existe.")
            st.session_state[f"{slot}_truth"] = "fake"
            log_entry(res, f"Generada ψ={entry['psi']}")

        if st.session_state.get(slot) is not None:
            res = st.session_state[slot]
            render_result(res, st.session_state.get(f"{slot}_prov"), "Generada", "gan",
                           st.session_state.get("meta_names"))
            # Sabemos con certeza que estas son sintéticas: señalar los fallos
            if res["p_fake"] <= engine.DECISION_THRESHOLD:
                st.markdown(
                    '<div class="warnbox"><b>El modelo se equivocó aquí.</b> Esta cara '
                    'sí fue generada, y aun así la dio por auténtica. Le ocurre con '
                    'menos del 1 % de los rostros más diversos: son los que más se '
                    'acercan a una fotografía real.</div>',
                    unsafe_allow_html=True)


def section_story():
    st.markdown("#### Cómo mira un Vision Transformer")
    st.markdown(
        '<div class="explain">Una red convolucional recorre la imagen con una lupa '
        'pequeña. Un <em>Vision Transformer</em> hace algo distinto: la corta en 196 '
        'cuadros y deja que cada cuadro le pregunte a todos los demás. Así puede notar '
        'que un arete no coincide con el otro, o que el fondo se retuerce junto al pelo '
        '— incoherencias entre zonas alejadas, que es justo donde los generadores '
        'se delatan.</div>', unsafe_allow_html=True)

    steps = [
        ("Preparar la entrada",
         "La imagen se lleva a 224 × 224 píxeles y se ajusta con las mismas medidas "
         "que usó la red cuando aprendió a ver, hace años, con millones de fotos."),
        ("Cortar en cuadros",
         "Se divide en 196 cuadros de 16 × 16. Cada uno se convierte en un vector, "
         "y se le añade su posición para que el modelo recuerde dónde estaba."),
        ("Dejar que se comparen",
         "Doce capas de atención permiten que cada cuadro consulte al resto. Un token "
         "especial va recogiendo la conclusión general."),
        ("Dar el veredicto",
         "Ese token pasa por una última capa que produce dos números: cuánto parece "
         "auténtica y cuánto parece fabricada."),
    ]
    html = '<div class="steps">'
    for i, (title, desc) in enumerate(steps, 1):
        html += (f'<div class="step-row"><div class="n">{i}</div><div>'
                 f'<div class="t">{title}</div><div class="d">{desc}</div></div></div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### La investigación")
    st.markdown(
        '<div class="explain">El primer modelo parecía un éxito. Acertaba el 82 % de '
        'las imágenes falsas del conjunto público. Hasta que le pedimos a StyleGAN3 '
        'caras recién hechas y acertó apenas el <em>23 %</em>. Algo estaba mal, y el '
        'modelo no era el culpable.</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="timeline">'
        '<div class="tl-item"><div class="step">Punto de partida</div>'
        '<div class="head">Un resultado demasiado bueno</div>'
        '<div class="body">El modelo alcanzaba <b>82 %</b> de aciertos sobre las falsas '
        'del conjunto de Kaggle. Parecía listo.</div></div>'
        '<div class="tl-item"><div class="step">La grieta</div>'
        '<div class="head">Caras nuevas, resultados pésimos</div>'
        '<div class="body">Al generar rostros frescos con el mismo StyleGAN3, la '
        'detección se desplomó a <b>23 %</b>. Peor que lanzar una moneda.</div></div>'
        '<div class="tl-item"><div class="step">Sospecha 1</div>'
        '<div class="head">¿Sería la compresión?</div>'
        '<div class="body">Medimos formato, peso por píxel y tablas de compresión de '
        'las dos clases. Todo idéntico: <b>PNG sin pérdida</b>. Descartado.</div></div>'
        '<div class="tl-item"><div class="step">Sospecha 2</div>'
        '<div class="head">¿Sería el redimensionado?</div>'
        '<div class="body">Probamos <b>seis formas</b> distintas de reducir la imagen. '
        'La detección se movió tres puntos. Descartado.</div></div>'
        '<div class="tl-item hit"><div class="step">Sospecha 3 · confirmada</div>'
        '<div class="head">Era el dial de diversidad</div>'
        '<div class="body">El conjunto público se había generado con la diversidad casi '
        'apagada. Todas sus caras falsas eran suaves y promediadas. El modelo aprendió '
        'a reconocer <b>eso</b>, no al generador.</div></div>'
        '<div class="tl-item hit"><div class="step">Corrección</div>'
        '<div class="head">Enseñarle todo el abanico</div>'
        '<div class="body">Generamos miles de caras cubriendo <b>toda</b> la escala de '
        'diversidad, y elegimos el mejor momento del entrenamiento según el escenario '
        'difícil. La detección subió a <b>87 %</b>, y afinando el entrenamiento '
        'llegamos a <b>94 %</b> con tres modelos promediados.</div></div>'
        '<div class="tl-item hit"><div class="step">El giro final</div>'
        '<div class="head">El modelo grande no era el bueno</div>'
        '<div class="body">Probamos dos arquitecturas distintas al transformador de '
        'parches. <b>ConvNeXt</b>, una red convolucional con un tercio de los '
        'parámetros, acertó el <b>99.6 %</b> por sí sola. Los rastros que deja el '
        'generador son texturas finas, y el transformador las promediaba al partir la '
        'imagen en cuadros de 16 píxeles.</div></div>'
        '<div class="tl-item hit"><div class="step">Resultado</div>'
        '<div class="head">99.7 % con la quinta parte del tamaño</div>'
        '<div class="body">Promediar ConvNeXt y Swin da <b>997 aciertos de cada '
        '1 000</b> usando 55 millones de parámetros, frente a los 258 millones de los '
        'tres transformadores que rendían 94 %.</div></div>'
        '</div>', unsafe_allow_html=True)

    st.markdown("##### La prueba que lo confirmó")
    st.plotly_chart(charts.truncation_chart(), use_container_width=True,
                     config={"displayModeBar": False})
    st.markdown('<div class="caption-mono">Cada punto resume 200 rostros. A la izquierda, '
                 'caras suaves que el modelo detectaba siempre. A la derecha, caras '
                 'diversas que se le escapaban casi todas.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### Descartando el redimensionado")
        st.plotly_chart(charts.resampling_chart(), use_container_width=True,
                         config={"displayModeBar": False})
    with c2:
        st.markdown("##### Verificando el generador")
        st.plotly_chart(charts.spectral_chart(), use_container_width=True,
                         config={"displayModeBar": False})
        st.markdown('<div class="caption-mono">Las falsas del conjunto se parecen más a '
                     'las nuestras que a las fotos reales: mismo generador, distinta '
                     'configuración.</div>', unsafe_allow_html=True)


def section_results(metrics, log_df):
    st.markdown("#### De 23 % a 99.7 %")
    st.markdown(
        '<div class="explain">Cinco iteraciones separan el primer intento del '
        'resultado final. Las dos primeras corrigieron los <em>datos</em>; la tercera, '
        'cómo se entrena; la última cambió la <em>arquitectura</em>, y fue la que más '
        'aportó.</div>', unsafe_allow_html=True)

    st.plotly_chart(charts.versions_chart(), use_container_width=True,
                     config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("#### El hallazgo que no esperábamos")
    st.markdown(
        '<div class="explain">El Vision Transformer es el modelo más grande y el que '
        'da nombre al proyecto. Resultó ser <em>el peor de los tres</em>. Una red '
        'convolucional con un tercio de sus parámetros acierta seis puntos más.'
        '<br><br>La explicación está en cómo mira cada uno: el transformador parte la '
        'imagen en cuadros de 16 × 16 píxeles y los compara entre sí, promediando lo '
        'que hay dentro de cada cuadro. Las huellas que deja StyleGAN3 son texturas '
        'muy finas — grano de piel, borde del cabello — y se pierden en ese promedio. '
        'Las convoluciones, en cambio, recorren la imagen con filtros pequeños y las '
        'conservan.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        st.plotly_chart(charts.architecture_chart(), use_container_width=True,
                         config={"displayModeBar": False})
        st.markdown('<div class="caption-mono">Arriba a la izquierda es mejor: más '
                     'aciertos con menos parámetros.</div>', unsafe_allow_html=True)
    with c2:
        st.plotly_chart(charts.ensemble_chart(), use_container_width=True,
                         config={"displayModeBar": False})
        st.markdown('<div class="caption-mono">Quitar los tres transformadores del '
                     'promedio mejoró el resultado y redujo el tamaño cinco veces.</div>',
                     unsafe_allow_html=True)

    if metrics:
        hard = metrics["evaluations"].get("test_hard", {}).get("tuned", {})
        easy = metrics["evaluations"].get("test", {}).get("tuned", {})

        st.markdown("---")
        st.markdown("#### Cómo se comporta hoy")
        st.markdown(
            '<div class="explain">Medido sobre mil rostros que el modelo nunca vio: '
            'quinientas fotografías reales y quinientas caras generadas con semillas '
            'nuevas, a máxima diversidad.</div>', unsafe_allow_html=True)

        cols = st.columns(4)
        cols[0].metric("Aciertos", f'{hard.get("accuracy", 0) * 100:.1f} %')
        cols[1].metric("Equilibrio (F1)", f'{hard.get("f1", 0):.4f}')
        cols[2].metric("Calidad (AUC)", f'{hard.get("auc_roc", 0):.4f}')
        cols[3].metric("Detecta sintéticas", f'{hard.get("recall_fake", 0) * 100:.1f} %')

        d1, d2 = st.columns(2, gap="large")
        with d1:
            st.markdown("##### Caras nuevas y diversas")
            f = charts.confusion_chart(hard.get("confusion_matrix"))
            if f:
                st.plotly_chart(f, use_container_width=True,
                                 config={"displayModeBar": False})
        with d2:
            st.markdown("##### Prueba estándar")
            f = charts.confusion_chart(easy.get("confusion_matrix"))
            if f:
                st.plotly_chart(f, use_container_width=True,
                                 config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("#### Qué significa este número, y qué no")
    st.markdown(
        '<div class="explain">Acertar 997 de cada 1 000 suena a resolver el problema, '
        'y conviene ser preciso sobre el alcance real.<br><br>'
        'El sistema distingue <b>rostros de StyleGAN3 frente a fotografías del '
        'conjunto FFHQ</b>. Eso no equivale a detectar cualquier imagen hecha por una '
        'inteligencia artificial: ante caras de Midjourney, Stable Diffusion o incluso '
        'de una versión anterior de StyleGAN, el rendimiento sería muy distinto y '
        'probablemente pobre — por la misma razón que la primera versión fallaba, '
        'porque aprende los rastros del generador concreto con el que se entrenó.'
        '<br><br>Con todo, el resultado se verificó dos veces: el pipeline completo se '
        'regeneró desde cero, con el dataset descargado de nuevo, y devolvió las '
        'mismas cifras hasta el cuarto decimal.</div>', unsafe_allow_html=True)

    if log_df is not None and not log_df.empty:
        st.markdown("---")
        st.markdown("#### El entrenamiento, época por época")
        st.markdown(
            '<div class="explain">La línea violeta mide el escenario difícil, y es la '
            'que decide cuándo parar. Se conservan los pesos de las tres mejores '
            'épocas promediados, no los de la última.</div>', unsafe_allow_html=True)
        f = charts.training_chart(log_df)
        if f:
            st.plotly_chart(f, use_container_width=True,
                             config={"displayModeBar": False})


def section_system():
    st.markdown("#### Repartir el trabajo")
    st.markdown(
        '<div class="explain">Procesar 20 000 fotos de un megapíxel cada una, una por '
        'una, toma varios minutos de espera muerta. Repartiendo el trabajo entre '
        '<em>16 procesos</em> del servidor, el mismo trabajo baja a segundos.</div>',
        unsafe_allow_html=True)

    st.plotly_chart(charts.preprocessing_chart(), use_container_width=True,
                     config={"displayModeBar": False})
    st.markdown('<div class="caption-mono">Decodificar, redimensionar y normalizar cada '
                 'imagen es independiente del resto, así que se puede repartir sin '
                 'coordinación entre procesos.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Las piezas del sistema")
    cards = [
        ("Inventario", "Polars", "Recorre las 20 000 imágenes y arma la tabla de "
         "metadatos en menos de un segundo, aprovechando todos los núcleos sin "
         "que haya que pedírselo."),
        ("Preprocesamiento", "16 procesos", "Cada proceso toma su lote de imágenes y "
         "las convierte en tensores. Sin esperas entre ellos."),
        ("Almacenamiento", "Parquet y uint8", "Guardar los píxeles crudos en lugar de "
         "números decimales redujo el espacio cuatro veces, decisivo con una cuota "
         "de 20 GB."),
        ("Entrenamiento", "NVIDIA L40S", "Ajustar 86 millones de parámetros con "
         "precisión mixta. Cada pasada completa por las 14 400 imágenes toma unos "
         "25 segundos."),
        ("Planificación", "SLURM", "Los trabajos se encolan y corren solos. Como cada "
         "uno tiene un límite de cuatro horas, guardan su progreso en cada época."),
        ("Generación", "StyleGAN3", "El generador de NVIDIA produce 36 rostros por "
         "segundo en la GPU del clúster. Cada uno se reproduce exactamente desde su "
         "semilla, así que no hace falta guardarlos."),
    ]
    for row in range(0, len(cards), 3):
        cols = st.columns(3, gap="medium")
        for col, (title, big, body) in zip(cols, cards[row:row + 3]):
            col.markdown(
                f'<div class="card"><div class="card-title">{title}</div>'
                f'<div class="card-big">{big}</div>'
                f'<div class="card-body">{body}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Lo que costó llegar aquí")
    obstacles = [
        ("La conexión directa al clúster estaba bloqueada",
         "Se trabajó desde la interfaz web del propio clúster, equivalente en la práctica."),
        ("La carpeta personal se llenó dos veces",
         "El proyecto se mudó al sistema de archivos grande, con cuatro veces más espacio."),
        ("Las particiones normales no tenían tarjeta gráfica",
         "Encontramos dos particiones con GPU que no aparecían en la documentación."),
        ("Una escritura quedó a medias por falta de espacio",
         "Se rediseñó el formato de guardado para ocupar la cuarta parte."),
        ("Cada trabajo se corta a las cuatro horas",
         "El entrenamiento guarda su avance en cada época y puede continuar donde quedó."),
        ("El modelo memorizaba en lugar de aprender",
         "Ritmo de aprendizaje cinco veces más lento y parada automática según el "
         "escenario difícil."),
        ("Una limpieza de disco borró los modelos entrenados",
         "El pipeline completo se regeneró desde cero en 35 minutos y devolvió las "
         "mismas cifras hasta el cuarto decimal, lo que dejó demostrada la "
         "reproducibilidad del sistema."),
    ]
    for prob, sol in obstacles:
        st.markdown(
            f'<div class="hyp out"><div class="row">'
            f'<span class="name">{prob}</span></div>'
            f'<div class="detail">{sol}</div></div>', unsafe_allow_html=True)


def footer():
    st.markdown('''
<div class="site-footer">
  <div class="footer-grid">
    <div>
      <h5>Equipo</h5>
      <ul>
        <li>Jason Barrantes Sánchez</li>
        <li>Melany Ramírez Anchía</li>
        <li>Walter Bowyer Carpenter</li>
        <li>Mauro Espinoza Hernández</li>
      </ul>
    </div>
    <div>
      <h5>Curso</h5>
      <p>Computación Paralela y Distribuida<br>
      Ingeniería en Ciencia de Datos<br>
      Universidad LEAD · San José, Costa Rica<br>
      Prof. Johansell Villalobos Cubillo</p>
    </div>
    <div>
      <h5>Cómputo</h5>
      <p>Clúster Kabré<br>
      Centro Nacional de Alta Tecnología<br>
      Partición nukwa-l40s · NVIDIA L40S</p>
    </div>
    <div>
      <h5>Construido con</h5>
      <ul>
        <li><a href="https://pytorch.org" target="_blank" rel="noopener">PyTorch</a> · BSD-3</li>
        <li><a href="https://github.com/NVlabs/stylegan3" target="_blank" rel="noopener">StyleGAN3</a> · NVIDIA, no comercial</li>
        <li><a href="https://pola.rs" target="_blank" rel="noopener">Polars</a> · MIT</li>
        <li><a href="https://streamlit.io" target="_blank" rel="noopener">Streamlit</a> · Apache 2.0</li>
        <li><a href="https://plotly.com/python/" target="_blank" rel="noopener">Plotly</a> · MIT</li>
      </ul>
    </div>
  </div>
  <div class="footer-legal">
    El clasificador parte de ViT-B/16 con pesos de ImageNet-1K distribuidos por
    torchvision. Los rostros sintéticos provienen de StyleGAN3-R entrenado sobre
    FFHQ, publicado por NVIDIA bajo licencia de uso no comercial; este trabajo es
    académico y no persigue fin comercial.<br><br>
    Las imágenes que subas o captures se procesan en tu propia máquina: no viajan
    a ningún servidor ni quedan guardadas.<br><br>
    Ninguna herramienta de detección es infalible. Toma el resultado como un
    indicio, nunca como una prueba.
  </div>
</div>''', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(theme.get_css(), unsafe_allow_html=True)
    if "log" not in st.session_state:
        st.session_state.log = []

    models, device, meta = get_model()
    metrics = get_metrics()
    log_df = get_training_log()
    pool = get_pool()

    st.session_state["meta_names"] = meta["names"] if meta else None
    hero(meta, metrics)

    if not models:
        st.error(
            "No se encuentran los modelos. Copia `vit_v5_cnx.pt` y "
            "`vit_v5_swin.pt` desde `checkpoints/` del clúster a la carpeta "
            "`assets/` de este dashboard.")
        footer()
        st.stop()

    t1, t2, t3, t4 = st.tabs(["Probar", "La investigación", "Resultados", "El sistema"])
    with t1:
        section_analyze(models, device, pool)
    with t2:
        section_story()
    with t3:
        section_results(metrics, log_df)
    with t4:
        section_system()

    if st.session_state.log:
        st.markdown("---")
        st.markdown("##### Lo analizado en esta sesión")
        df = pd.DataFrame(st.session_state.log)
        st.dataframe(df, use_container_width=True, hide_index=True)
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            st.download_button("Descargar", df.to_csv(index=False).encode(),
                                "analisis_sesion.csv", "text/csv",
                                use_container_width=True)
        with c2:
            if st.button("Vaciar", use_container_width=True, type="secondary"):
                st.session_state.log = []
                st.rerun()

    footer()


if __name__ == "__main__":
    main()
