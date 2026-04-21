import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from astropy import units as u
from astropy.constants import G, c

# --- 1. НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Гравитационная линза", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #05070a; color: #e0e0e0; }
    .stMetric { border: 1px solid #1f2937; background-color: #111827; border-radius: 8px; padding: 10px; }
    [data-testid="stSidebar"] { background-color: #111827; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ДВИЖОК РЕНДЕРИНГА (МОДЕЛЬ SIE) ---
def render_lensing_final(img, theta_e_px, q, lx, ly):
    h, w = img.shape[:2]
    y, x = np.indices((h, w))
    dy, dx = y - (h//2 + ly), x - (w//2 + lx)
    
    # Модель эллиптического потенциала (SIE)
    r_ell = np.sqrt(q * dx**2 + dy**2 / q + 1e-9)
    
    # Уравнение линзы: расчет угла отклонения
    scale = (theta_e_px**2) / r_ell
    src_x = x - dx * (scale / r_ell) * q
    src_y = y - dy * (scale / r_ell) / q
    
    out = np.zeros_like(img)
    for i in range(3):
        out[:,:,i] = map_coordinates(img[:,:,i], [src_y, src_x], order=1, mode='constant', cval=0)
    return out

# --- 3. ГЕНЕРАТОР КОСМИЧЕСКИХ ОБЪЕКТОВ ---
def generate_source_pro(type, temp):
    size = 600
    img = np.zeros((size, size, 3), dtype=np.float32)
    y, x = np.indices((size, size))
    r = np.sqrt((x-300)**2 + (y-300)**2)
    phi = np.arctan2(y-300, x-300)
    
    # Температурные палитры звезд
    palettes = {
        "Стандартный": [1.0, 0.95, 0.9], 
        "Горячий (Голубой)": [0.7, 0.9, 1.5], 
        "Холодный (Красный)": [1.4, 0.4, 0.1]
    }
    rgb = palettes[temp]
    
    if type == "Спиральная галактика":
        spiral = np.cos(2 * phi - 0.5 * np.sqrt(r))
        mask = 255 * np.exp(-r/30) + 180 * np.exp(-r/90) * (spiral**2)
    elif type == "Квазар":
        mask = 255 * np.exp(-r/7) + 60 * np.exp(-r/45)
    elif type == "Двойной квазар":
        r1 = np.sqrt((x-315)**2 + (y-300)**2)
        r2 = np.sqrt((x-285)**2 + (y-300)**2)
        mask = 255 * np.exp(-r1/6) + 255 * np.exp(-r2/6) + 40 * np.exp(-r/50)
    else: # Эллиптическая галактика
        mask = 255 * np.exp(-r/45)
        
    for i in range(3):
        img[:,:,i] = np.clip(mask * rgb[i], 0, 255)
    
    stars = np.random.random((size, size)) > 0.998
    img[stars] = 230
    return img.astype(np.uint8)

# --- 4. ПАНЕЛЬ УПРАВЛЕНИЯ (SIDEBAR) ---
st.sidebar.title("⚙️ ПАРАМЕТРЫ")
st.sidebar.markdown("---")

lens_obj = st.sidebar.selectbox("Объект-линза", ["Черная дыра", "Галактика", "Скопление"])
lens_mass_map = {"Черная дыра": 12.0, "Галактика": 13.8, "Скопление": 15.5}

preset = st.sidebar.selectbox("Сценарии ОТО", ["Ручной режим", "Крест Эйнштейна", "Кольцо Эйнштейна", "Линзирование квазара"])

if preset == "Крест Эйнштейна":
    q, m_log, lx, ly = 0.52, 13.2, 12, 0
elif preset == "Кольцо Эйнштейна":
    q, m_log, lx, ly = 1.0, 13.8, 0, 0
elif preset == "Линзирование квазара":
    q, m_log, lx, ly = 0.88, 13.5, 45, 20
else:
    m_log = st.sidebar.slider("Масса объекта (log10 M☉)", 10.0, 17.0, lens_mass_map[lens_obj])
    q = st.sidebar.slider("Сплюснутость (q)", 0.1, 1.0, 1.0)
    lx = st.sidebar.slider("Смещение X", -250, 250, 0)
    ly = st.sidebar.slider("Смещение Y", -250, 250, 0)

st.sidebar.markdown("---")
dist_l = st.sidebar.slider("Расстояние до линзы (Мпк)", 100, 5000, 1000)
dist_s = st.sidebar.slider("Расстояние до источника (Мпк)", 5100, 20000, 9000)

st.sidebar.markdown("---")
src_type = st.sidebar.selectbox("Тип источника", ["Спиральная галактика", "Квазар", "Двойной квазар", "Эллиптическая галактика"])
src_temp = st.sidebar.radio("Температура (Спектр)", ["Стандартный", "Горячий (Голубой)", "Холодный (Красный)"])

# --- 5. ФИЗИЧЕСКИЕ РАСЧЕТЫ ---
M = (10**m_log) * u.solMass
Dl, Ds = dist_l * u.Mpc, dist_s * u.Mpc
Dls = Ds - Dl
theta_e_rad = np.sqrt((4*G*M / c**2) * (Dls / (Dl * Ds)))
theta_e_arc = (theta_e_rad * u.rad).to(u.arcsec).value
type_mults = {"Черная дыра": 45, "Галактика": 65, "Скопление": 110}
t_px = theta_e_arc * type_mults[lens_obj]

# --- 6. ИНТЕРФЕЙС (ВКЛАДКИ) ---
st.title("Система визуализации гравитационного линзирования")

tab_obs, tab_geo = st.tabs(["🚀 НАБЛЮДЕНИЕ", "📐 ГЕОМЕТРИЧЕСКАЯ СХЕМА"])

with tab_obs:
    source_img = generate_source_pro(src_type, src_temp)
    result_img = render_lensing_final(source_img, t_px, q, lx, ly)
    
    # Маркер линзы
    cy, cx = 300 + ly, 300 + lx
    if 0 <= cy < 600 and 0 <= cx < 600:
        l_color = {"Черная дыра": [255, 140, 0], "Галактика": [255, 255, 240], "Скопление": [0, 242, 255]}[lens_obj]
        result_img[cy-4:cy+4, cx-4:cx+4] = l_color

    st.image(result_img, width=800)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Радиус Эйнштейна", f"{theta_e_arc:.4f}\"")
    c2.metric("Отношение расстояний", f"{(Dls/Ds).value:.2f}")
    c3.write(f"**Источник:** {src_temp}")

with tab_geo:
    st.subheader("Ход световых лучей (Вид сверху)")
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#05070a')
    ax.set_facecolor('#05070a')
    
    # Схема: Наблюдатель (0) --- Линза (dist_l) --- Источник (dist_s)
    z_p = [0, dist_l, dist_s]
    beam = theta_e_arc * 0.12
    
    ax.plot(z_p, [0, beam, 0], color='#00f2ff', lw=2, label="Путь света")
    ax.plot(z_p, [0, -beam, 0], color='#00f2ff', lw=2)
    ax.plot([0, dist_l], [0, beam], 'w--', alpha=0.3)
    ax.plot([0, dist_l], [0, -beam], 'w--', alpha=0.3)
    
    ax.scatter(z_p, [0,0,0], c=['white', '#ff8c00', '#1e90ff'], s=[80, 250, 100], zorder=3)
    ax.text(0, -0.15, "Наблюдатель", color='white', ha='center')
    ax.text(dist_l, -0.25, f"Линза ({lens_obj})", color='#ff8c00', ha='center')
    ax.text(dist_s, -0.15, "Источник", color='#1e90ff', ha='center')
    
    ax.set_ylim(-1, 1); ax.axis('off')
    st.pyplot(fig)
    st.latex(r"\theta_E = \sqrt{\frac{4GM}{c^2} \frac{D_{ls}}{D_l D_s}}")
    st.info("Схема наглядно показывает отклонение лучей в соответствии с уравнениями Общей Теории Относительности.")
