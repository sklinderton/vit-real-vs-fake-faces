"""
theme.py — Sistema de diseño (tema oscuro)
===========================================
Requiere `.streamlit/config.toml` con base="dark". Sin ese archivo los
componentes internos de Streamlit quedan descoordinados con este CSS.

Color: teal para auténtica, violeta para sintética, ambos elevados en
luminosidad para leer bien sobre fondo oscuro. Se evita el par verde/rojo
porque impone una lectura moral que no corresponde y porque teal/violeta
se distingue en los tipos más frecuentes de daltonismo.
"""

# ── Superficies ───────────────────────────────────────────────────────────
BG = "#0F141A"
SURFACE = "#171E27"
SURFACE_2 = "#1E2733"
SURFACE_3 = "#26313F"

# ── Texto ─────────────────────────────────────────────────────────────────
TEXT = "#E8EDF3"
TEXT_DIM = "#95A6B8"
TEXT_FAINT = "#6B7C8F"
RULE = "#2A3542"
RULE_SOFT = "#212B36"

# ── Señal ─────────────────────────────────────────────────────────────────
AUTHENTIC = "#2DD4A7"
AUTHENTIC_DIM = "#1A7C63"
SYNTHETIC = "#A78BFA"
SYNTHETIC_DIM = "#6D4DBF"
BLUE = "#5AA9FF"
AMBER = "#F0B429"

FONT_DISPLAY = "'Archivo', system-ui, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, monospace"
PLOTLY_FONT = "Archivo, system-ui, sans-serif"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ══ Base ══════════════════════════════════════════════════════════════ */
.stApp { background: __BG__; }
html, body, [class*="css"], .stMarkdown, p, li, span, div, label {
    font-family: __FONT_DISPLAY__; color: __TEXT__;
}
h1,h2,h3,h4,h5 { font-family:__FONT_DISPLAY__ !important; color:__TEXT__ !important;
    letter-spacing:-0.025em; font-weight:700; }
.block-container { padding-top:1.6rem; padding-bottom:0; max-width:1200px; }
footer, #MainMenu, header[data-testid="stHeader"] { visibility:hidden; height:0; }
hr { border-color:__RULE__; margin:2rem 0; }

/* ══ Animaciones ═══════════════════════════════════════════════════════ */
@keyframes riseIn { from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;} }
@keyframes pulseDot { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.rise { animation: riseIn 0.5s cubic-bezier(.2,.7,.3,1) both; }
.rise-2 { animation-delay:0.1s; } .rise-3 { animation-delay:0.18s; }

/* ══ Portada ═══════════════════════════════════════════════════════════ */
.hero { padding:0.4rem 0 0.2rem 0; }
.eyebrow {
    font-family:__FONT_MONO__; font-size:0.66rem; letter-spacing:0.2em;
    text-transform:uppercase; color:__TEXT_DIM__; margin-bottom:0.75rem;
    display:flex; align-items:center; gap:0.55rem;
}
.eyebrow::before { content:''; width:7px; height:7px; border-radius:50%;
    background:__SYNTHETIC__; animation:pulseDot 2.6s ease-in-out infinite; }
.hero h1 { font-size:3.2rem; font-weight:900; line-height:1.0;
    letter-spacing:-0.045em; margin:0 0 0.75rem 0; }
.hero h1 .accent {
    background: linear-gradient(100deg, __SYNTHETIC__ 10%, __BLUE__ 55%, __AUTHENTIC__ 95%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero .lede { font-size:1.05rem; color:__TEXT_DIM__; max-width:60ch; line-height:1.62; margin:0; }

/* ══ Franja de datos ═══════════════════════════════════════════════════ */
.datastrip { display:flex; flex-wrap:wrap; border:1px solid __RULE__; border-radius:6px;
    background:__SURFACE__; margin:1.7rem 0 0.5rem 0; overflow:hidden; }
.datastrip .cell { flex:1 1 130px; padding:0.9rem 1.1rem; border-right:1px solid __RULE__; }
.datastrip .cell:last-child { border-right:none; }
.datastrip .k { font-family:__FONT_MONO__; font-size:0.57rem; letter-spacing:0.15em;
    text-transform:uppercase; color:__TEXT_FAINT__; display:block; margin-bottom:0.3rem; }
.datastrip .v { font-family:__FONT_MONO__; font-size:1.02rem; font-weight:600; color:__TEXT__; }
.datastrip .v.hl { color:__SYNTHETIC__; }

/* ══ Pestañas ══════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] { gap:0.3rem; border-bottom:1px solid __RULE__;
    margin-bottom:1.7rem; background:transparent; }
.stTabs [data-baseweb="tab"] { font-family:__FONT_MONO__; font-size:0.72rem;
    letter-spacing:0.09em; text-transform:uppercase; font-weight:500;
    background:transparent; border-radius:0; padding:0.75rem 1.1rem;
    color:__TEXT_FAINT__; transition:color 0.15s ease; }
.stTabs [data-baseweb="tab"]:hover { color:__TEXT__; }
.stTabs [aria-selected="true"] { color:__TEXT__ !important;
    border-bottom:2px solid __SYNTHETIC__; font-weight:600; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

/* ══ Radio segmentado ══════════════════════════════════════════════════ */
div[role="radiogroup"] { gap:0.25rem !important; background:__SURFACE__;
    padding:0.3rem; border-radius:7px; border:1px solid __RULE__;
    display:inline-flex !important; flex-wrap:wrap; }
div[role="radiogroup"] > label { background:transparent !important; border-radius:5px;
    padding:0.5rem 1.15rem !important; margin:0 !important; cursor:pointer;
    transition:all 0.16s ease; border:1px solid transparent; }
div[role="radiogroup"] > label:hover { background:__SURFACE_2__ !important; }
div[role="radiogroup"] > label > div:first-child { display:none !important; }
div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
    font-family:__FONT_MONO__ !important; font-size:0.73rem !important;
    letter-spacing:0.04em; font-weight:500; margin:0 !important;
    color:__TEXT_DIM__ !important; transition:color 0.16s ease; }
div[role="radiogroup"] > label:has(input:checked) {
    background:__SYNTHETIC__ !important; border-color:__SYNTHETIC__;
    box-shadow:0 2px 10px rgba(167,139,250,0.28); }
div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
    color:#0F141A !important; font-weight:700; }

/* ══ Botones — contraste forzado en todos los hijos ════════════════════ */
.stButton > button, .stDownloadButton > button {
    font-family:__FONT_MONO__ !important; font-size:0.74rem !important;
    letter-spacing:0.07em; text-transform:uppercase; font-weight:600;
    border-radius:6px; border:1px solid __SYNTHETIC__;
    background:__SYNTHETIC__ !important; padding:0.55rem 1.3rem;
    transition:all 0.16s ease; }
.stButton > button *, .stDownloadButton > button *,
.stButton > button p, .stDownloadButton > button p,
.stButton > button div, .stDownloadButton > button div,
.stButton > button span, .stDownloadButton > button span {
    color:#0F141A !important; font-weight:600 !important; }
.stButton > button:hover, .stDownloadButton > button:hover {
    background:#BFA6FF !important; border-color:#BFA6FF;
    transform:translateY(-1px); box-shadow:0 6px 18px rgba(167,139,250,0.32); }
.stButton > button:active { transform:none; }

/* Variante secundaria */
.stButton > button[kind="secondary"] {
    background:__SURFACE_2__ !important; border-color:__RULE__; }
.stButton > button[kind="secondary"] * { color:__TEXT__ !important; }
.stButton > button[kind="secondary"]:hover {
    background:__SURFACE_3__ !important; border-color:__TEXT_FAINT__; }

/* ══ Uploader ══════════════════════════════════════════════════════════ */
section[data-testid="stFileUploaderDropzone"] {
    background:__SURFACE__ !important; border:1.5px dashed __RULE__ !important;
    border-radius:7px; transition:all 0.16s ease; }
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color:__SYNTHETIC__ !important; background:__SURFACE_2__ !important; }
section[data-testid="stFileUploaderDropzone"] span,
section[data-testid="stFileUploaderDropzone"] small,
section[data-testid="stFileUploaderDropzone"] div { color:__TEXT_DIM__ !important; }
section[data-testid="stFileUploaderDropzone"] button {
    background:__SYNTHETIC__ !important; border:none !important; border-radius:5px;
    font-family:__FONT_MONO__ !important; font-size:0.71rem !important;
    letter-spacing:0.05em; text-transform:uppercase; }
section[data-testid="stFileUploaderDropzone"] button * { color:#0F141A !important; font-weight:600 !important; }
div[data-testid="stFileUploaderFile"] { background:__SURFACE_2__; border-radius:5px; }
div[data-testid="stFileUploaderFile"] * { color:__TEXT__ !important; }

/* ══ Cámara ════════════════════════════════════════════════════════════ */
div[data-testid="stCameraInput"] button {
    background:__SYNTHETIC__ !important; border:none !important; }
div[data-testid="stCameraInput"] button * { color:#0F141A !important; font-weight:600 !important; }

/* ══ Selectbox ═════════════════════════════════════════════════════════ */
/* BaseWeb monta la lista desplegable en un portal al final del <body>, fuera
   del árbol de la aplicación. Por eso no basta con anidar reglas bajo .stApp:
   hay que apuntar a los atributos del portal y forzar el color en todos los
   descendientes, porque el texto va envuelto en varios divs. Se cubren varias
   combinaciones de selector porque cambian entre versiones de Streamlit. */

/* Control cerrado */
div[data-baseweb="select"],
div[data-baseweb="select"] > div,
div[data-baseweb="select"] [role="combobox"] {
    background:__SURFACE__ !important;
    border-color:__RULE__ !important;
    border-radius:6px;
}
div[data-baseweb="select"] *:not(svg):not(path) {
    font-family:__FONT_MONO__ !important;
    font-size:0.82rem !important;
    color:__TEXT__ !important;
}
div[data-baseweb="select"] > div:hover { border-color:__SYNTHETIC__ !important; }
div[data-baseweb="select"] svg { fill:__TEXT_DIM__ !important; }

/* Contenedor de la lista desplegada */
div[data-baseweb="popover"] div[data-baseweb="menu"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] > div > div,
ul[data-baseweb="menu"],
div[role="listbox"] {
    background:__SURFACE_2__ !important;
    border:1px solid __RULE__ !important;
    border-radius:6px !important;
    box-shadow:0 12px 34px rgba(0,0,0,0.55) !important;
}

/* Cada opción */
div[data-baseweb="popover"] li,
ul[data-baseweb="menu"] li,
div[role="listbox"] li,
li[role="option"] {
    background:__SURFACE_2__ !important;
    font-family:__FONT_MONO__ !important;
    font-size:0.82rem !important;
    color:__TEXT__ !important;
}

/* El texto suele ir dentro de divs anidados: se fuerza en todo descendiente */
div[data-baseweb="popover"] li *,
ul[data-baseweb="menu"] li *,
div[role="listbox"] li *,
li[role="option"] * {
    color:__TEXT__ !important;
    background:transparent !important;
}

/* Opción bajo el cursor o resaltada con el teclado */
div[data-baseweb="popover"] li:hover,
ul[data-baseweb="menu"] li:hover,
li[role="option"]:hover,
li[role="option"][data-highlighted="true"] {
    background:__SURFACE_3__ !important;
}
div[data-baseweb="popover"] li:hover *,
ul[data-baseweb="menu"] li:hover *,
li[role="option"]:hover *,
li[role="option"][data-highlighted="true"] * {
    background:transparent !important;
}

/* Opción ya seleccionada */
li[role="option"][aria-selected="true"] {
    background:__SYNTHETIC__ !important;
}
li[role="option"][aria-selected="true"] * {
    color:#0F141A !important;
    font-weight:600 !important;
    background:transparent !important;
}

/* ══ Toggle ════════════════════════════════════════════════════════════ */
div[data-testid="stCheckbox"] label p, div[data-testid="stToggle"] label p {
    font-family:__FONT_MONO__ !important; font-size:0.73rem !important;
    color:__TEXT_DIM__ !important; letter-spacing:0.03em; }

/* ══ Tarjetas ══════════════════════════════════════════════════════════ */
.card { background:__SURFACE__; border:1px solid __RULE__; border-radius:7px;
    padding:1.3rem 1.45rem; margin-bottom:0.9rem; height:100%;
    transition:all 0.2s ease; }
.card:hover { border-color:__SURFACE_3__; transform:translateY(-2px);
    box-shadow:0 8px 24px rgba(0,0,0,0.28); }
.card-title { font-family:__FONT_MONO__; font-size:0.59rem; letter-spacing:0.16em;
    text-transform:uppercase; color:__TEXT_FAINT__; margin-bottom:0.65rem; }
.card-body { font-size:0.89rem; line-height:1.62; color:__TEXT_DIM__; }
.card-big { font-family:__FONT_MONO__; font-size:1.85rem; font-weight:600;
    letter-spacing:-0.02em; line-height:1; margin:0.15rem 0 0.4rem 0; color:__TEXT__; }

/* ══ Veredicto ═════════════════════════════════════════════════════════ */
.verdict-word { font-family:__FONT_DISPLAY__; font-size:2.9rem; font-weight:900;
    letter-spacing:-0.04em; line-height:1; margin:0.15rem 0 0.5rem 0;
    animation:riseIn 0.45s cubic-bezier(.2,.7,.3,1) both; }
.readout-grid { display:grid; grid-template-columns:1fr auto; gap:0.42rem 1.2rem;
    font-family:__FONT_MONO__; font-size:0.84rem; margin-top:1.1rem;
    padding-top:0.9rem; border-top:1px solid __RULE__; }
.readout-grid .k { color:__TEXT_DIM__; font-size:0.73rem; }
.readout-grid .v { font-weight:600; text-align:right; color:__TEXT__; }
.provenance { font-family:__FONT_MONO__; font-size:0.71rem; line-height:1.7;
    color:__TEXT_DIM__; background:__SURFACE_2__; border-left:3px solid __SYNTHETIC__;
    padding:0.75rem 0.95rem; margin-top:1.1rem; border-radius:0 5px 5px 0; }

/* ══ Aviso de falibilidad ══════════════════════════════════════════════ */
.warnbox { background:rgba(240,180,41,0.08); border:1px solid rgba(240,180,41,0.3);
    border-radius:6px; padding:0.85rem 1.05rem; margin-top:1rem;
    font-size:0.84rem; line-height:1.6; color:__TEXT_DIM__; }
.warnbox b { color:__AMBER__; }

/* ══ Métricas ══════════════════════════════════════════════════════════ */
div[data-testid="stMetric"] { background:__SURFACE__; border:1px solid __RULE__;
    border-radius:7px; padding:0.95rem 1.1rem; transition:border-color 0.2s ease; }
div[data-testid="stMetric"]:hover { border-color:__SURFACE_3__; }
div[data-testid="stMetricLabel"] p { font-family:__FONT_MONO__ !important;
    font-size:0.58rem !important; letter-spacing:0.14em; text-transform:uppercase;
    color:__TEXT_FAINT__ !important; }
div[data-testid="stMetricValue"] { font-family:__FONT_MONO__ !important;
    font-size:1.5rem !important; font-weight:600; color:__TEXT__ !important; }

/* ══ Texto ═════════════════════════════════════════════════════════════ */
.explain { font-size:0.96rem; line-height:1.72; color:__TEXT_DIM__; max-width:68ch; }
.explain em { color:__SYNTHETIC__; font-style:normal; font-weight:600; }
.explain b { color:__TEXT__; }
.note { font-size:0.87rem; line-height:1.64; color:__TEXT_DIM__;
    border-left:2px solid __RULE__; padding-left:0.95rem; margin:1rem 0; max-width:66ch; }
.caption-mono { font-family:__FONT_MONO__; font-size:0.68rem; color:__TEXT_FAINT__;
    line-height:1.65; margin-top:0.45rem; }

/* ══ Cronología ════════════════════════════════════════════════════════ */
.timeline { position:relative; padding-left:1.9rem; margin:1.4rem 0; }
.timeline::before { content:''; position:absolute; left:0.42rem; top:0.4rem;
    bottom:0.4rem; width:2px; background:linear-gradient(__RULE__, __SYNTHETIC__); }
.tl-item { position:relative; margin-bottom:1.5rem; }
.tl-item::before { content:''; position:absolute; left:-1.62rem; top:0.35rem;
    width:11px; height:11px; border-radius:50%; background:__BG__;
    border:2.5px solid __TEXT_FAINT__; }
.tl-item.hit::before { border-color:__SYNTHETIC__; background:__SYNTHETIC__; }
.tl-item .step { font-family:__FONT_MONO__; font-size:0.59rem; letter-spacing:0.14em;
    text-transform:uppercase; color:__TEXT_FAINT__; margin-bottom:0.28rem; }
.tl-item .head { font-size:1.02rem; font-weight:700; margin-bottom:0.32rem;
    letter-spacing:-0.015em; color:__TEXT__; }
.tl-item .body { font-size:0.89rem; line-height:1.64; color:__TEXT_DIM__; max-width:64ch; }
.tl-item .body b { color:__TEXT__; }

/* ══ Hipótesis ═════════════════════════════════════════════════════════ */
.hyp { background:__SURFACE__; border:1px solid __RULE__; border-radius:7px;
    padding:1rem 1.2rem; margin-bottom:0.65rem; border-left-width:3px;
    transition:transform 0.18s ease; }
.hyp:hover { transform:translateX(3px); }
.hyp.out { border-left-color:__TEXT_FAINT__; }
.hyp.in { border-left-color:__SYNTHETIC__; background:rgba(167,139,250,0.07); }
.hyp .row { display:flex; justify-content:space-between; align-items:baseline;
    gap:1rem; flex-wrap:wrap; }
.hyp .name { font-weight:600; font-size:0.95rem; color:__TEXT__; }
.hyp .tag { font-family:__FONT_MONO__; font-size:0.61rem; letter-spacing:0.11em;
    text-transform:uppercase; font-weight:600; padding:0.18rem 0.55rem;
    border-radius:4px; white-space:nowrap; }
.hyp.out .tag { color:__TEXT_FAINT__; background:__SURFACE_2__; }
.hyp.in .tag { color:#0F141A; background:__SYNTHETIC__; }
.hyp .detail { font-size:0.86rem; color:__TEXT_DIM__; margin-top:0.45rem; line-height:1.6; }

/* ══ Pasos ═════════════════════════════════════════════════════════════ */
.steps { display:grid; gap:0.5rem; margin:1.2rem 0; }
.step-row { display:grid; grid-template-columns:2.2rem 1fr; gap:1rem;
    align-items:start; background:__SURFACE__; border:1px solid __RULE__;
    border-radius:7px; padding:0.9rem 1.1rem; transition:border-color 0.18s ease; }
.step-row:hover { border-color:__SYNTHETIC__; }
.step-row .n { font-family:__FONT_MONO__; font-size:0.73rem; font-weight:700;
    color:#0F141A; background:__SYNTHETIC__; border-radius:5px;
    text-align:center; padding:0.24rem 0; }
.step-row .t { font-weight:650; font-size:0.94rem; margin-bottom:0.2rem; color:__TEXT__; }
.step-row .d { font-size:0.87rem; color:__TEXT_DIM__; line-height:1.6; }

/* ══ Tablas ════════════════════════════════════════════════════════════ */
div[data-testid="stDataFrame"] { border:1px solid __RULE__; border-radius:7px; }

/* ══ Footer ════════════════════════════════════════════════════════════ */
.site-footer { border-top:1px solid __RULE__; margin-top:3.2rem; padding:1.9rem 0 2.5rem 0; }
.footer-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(205px,1fr)); gap:1.9rem; }
.footer-grid h5 { font-family:__FONT_MONO__ !important; font-size:0.59rem !important;
    letter-spacing:0.16em; text-transform:uppercase; color:__TEXT_FAINT__ !important;
    margin:0 0 0.7rem 0; font-weight:500 !important; }
.footer-grid p, .footer-grid li { font-size:0.84rem; line-height:1.75;
    margin:0; color:__TEXT_DIM__; }
.footer-grid ul { list-style:none; padding:0; margin:0; }
.footer-grid a { color:__TEXT__; text-decoration:underline; text-underline-offset:2px;
    text-decoration-color:__RULE__; transition:color 0.15s ease; }
.footer-grid a:hover { color:__SYNTHETIC__; text-decoration-color:__SYNTHETIC__; }
.footer-legal { font-family:__FONT_MONO__; font-size:0.66rem; color:__TEXT_FAINT__;
    line-height:1.85; margin-top:1.8rem; padding-top:1.1rem; border-top:1px solid __RULE__; }

/* ══ Accesibilidad ═════════════════════════════════════════════════════ */
*:focus-visible { outline:2px solid __SYNTHETIC__; outline-offset:2px; }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation:none !important; transition:none !important; }
}
@media (max-width:720px) {
    .hero h1 { font-size:2.2rem; } .verdict-word { font-size:2.2rem; }
}
</style>
"""

_TOKENS = [
    ("__BG__", BG), ("__SURFACE_3__", SURFACE_3), ("__SURFACE_2__", SURFACE_2),
    ("__SURFACE__", SURFACE), ("__TEXT_DIM__", TEXT_DIM), ("__TEXT_FAINT__", TEXT_FAINT),
    ("__TEXT__", TEXT), ("__RULE_SOFT__", RULE_SOFT), ("__RULE__", RULE),
    ("__AUTHENTIC_DIM__", AUTHENTIC_DIM), ("__AUTHENTIC__", AUTHENTIC),
    ("__SYNTHETIC_DIM__", SYNTHETIC_DIM), ("__SYNTHETIC__", SYNTHETIC),
    ("__BLUE__", BLUE), ("__AMBER__", AMBER),
    ("__FONT_DISPLAY__", FONT_DISPLAY), ("__FONT_MONO__", FONT_MONO),
]


def get_css() -> str:
    css = CSS
    for token, value in _TOKENS:
        css = css.replace(token, value)
    return css
