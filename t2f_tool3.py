import sys
import os
import math
import numpy as np
from scipy.io import loadmat

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSlider, QLabel, QPushButton, QDoubleSpinBox, QGroupBox, QFrame,
    QComboBox, QListWidget, QFileDialog, QFormLayout, QTabWidget, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg

# =========================================================
# 📚 CONSTANTES DE CURVAS (IEC / IEEE)
# =========================================================
CURVES = {
    "IEC Standard Inverse":  (0.14, 0.0, 0.02),
    "IEC Very Inverse":      (13.5, 0.0, 1.0),
    "IEC Extremely Inverse": (80.0, 0.0, 2.0),
    "IEC Long Time Inverse": (120.0, 0.0, 1.0),
    "IEEE Moderately Inv":   (0.0515, 0.114, 0.02),
    "IEEE Very Inverse":     (19.61, 0.491, 2.0),
    "IEEE Extremely Inv":    (28.2, 0.1217, 2.0)
}

# =========================================================
# 🎨 ESTILOS (Dark Lab)
# =========================================================
THEME = {
    "bg_app": "#1e1e1e", "bg_panel": "#252526", "bg_widget": "#333333",
    "border": "#3e3e42", "text_main": "#e0e0e0", "text_dim": "#858585",
    "accent": "#007acc", "accent_hover": "#0098ff",
    "A": "#00ffff", "B": "#ff3333", "C": "#00ff00", 
    "V1": "#4facfe", "V2": "#f093fb", "V0": "#fcc203",
    "Curve": "#ffff00", 
    "cursor": "#FFFF00" 
}

pg.setConfigOption('background', THEME["bg_panel"])
pg.setConfigOption('foreground', THEME["text_dim"])
pg.setConfigOptions(antialias=True)

# =========================================================
# 🛠️ FUNÇÕES MATEMÁTICAS
# =========================================================
def load_matlab_file(path):
    try: return loadmat(path, squeeze_me=True, struct_as_record=False)
    except: return None

def _unwrap(x):
    if isinstance(x, np.ndarray):
        if x.size == 1:
            if x.dtype == object or x.ndim > 1:
                return _unwrap(x.item())
    return x

def recursive_find_signals(data, parent_key="", found_signals=None):
    if found_signals is None: found_signals = {}
    if isinstance(data, dict):
        for key, val in data.items():
            if key.startswith('__'): continue
            recursive_find_signals(val, key, found_signals)
    elif hasattr(data, '_fieldnames'):
        fields = data._fieldnames
        fields_lower = [f.lower() for f in fields]
        if 'time' in fields_lower and 'data' in fields_lower:
            try:
                t_key = fields[fields_lower.index('time')]
                d_key = fields[fields_lower.index('data')]
                t_val = getattr(data, t_key); d_val = getattr(data, d_key)
                t_arr = np.array(_unwrap(t_val), dtype=float).flatten()
                d_arr = np.array(_unwrap(d_val), dtype=float)
                if d_arr.ndim == 2:
                    if d_arr.shape[0] < d_arr.shape[1] and d_arr.shape[0] <= 4: d_arr = d_arr.T
                clean_name = parent_key.replace("ts_", "")
                found_signals[clean_name] = (t_arr, d_arr)
            except: pass
        else:
            for f in fields: recursive_find_signals(getattr(data, f), f"{parent_key}_{f}", found_signals)
    elif isinstance(data, np.ndarray) and data.dtype == object:
        for i, item in enumerate(data.flat): recursive_find_signals(item, f"{parent_key}", found_signals)
    return found_signals

def clarke_transform(a, b, c):
    k = math.sqrt(2/3)
    alpha = k * (a - 0.5*b - 0.5*c)
    beta  = k * ((math.sqrt(3)/2)*b - (math.sqrt(3)/2)*c)
    return alpha, beta

def true_rms(x):
    if len(x) == 0: return 0.0
    return np.sqrt(np.mean(np.square(x)))

def phasor_unit(x, t, f0=60.0):
    x = np.asarray(x); t = np.asarray(t)
    if len(x) < 2: return 0j
    x = x - np.mean(x)
    X = np.sum(x * np.exp(-1j * 2*np.pi*f0*(t - t[0])))
    if abs(X) < 1e-9: return 0j
    return X / abs(X)

def sym_components(Fa, Fb, Fc):
    a = np.exp(1j * 2*np.pi/3)
    F0 = (1/3) * (Fa + Fb + Fc)
    F1 = (1/3) * (Fa + a*Fb + a**2*Fc)
    F2 = (1/3) * (Fa + a**2*Fb + a*Fc)
    return F0, F1, F2

def calculate_tcc(I_val, Ip, TD, curve_name):
    if curve_name not in CURVES:
        A, B, p = CURVES["IEC Standard Inverse"]
    else:
        A, B, p = CURVES[curve_name]

    safe_Ip = Ip if Ip > 0 else 0.001
    
    # Vetorial
    if isinstance(I_val, np.ndarray):
        M = I_val / safe_Ip
        t = np.full_like(I_val, np.inf)
        mask = M > 1.001
        if np.any(mask):
            denom = np.power(M[mask], p) - 1
            denom[denom == 0] = 1e-9
            t[mask] = TD * ( (A / denom) + B )
        return t
    
    # Escalar
    else:
        M = I_val / safe_Ip
        if M <= 1.001: return float('inf')
        denom = (M**p - 1)
        if denom == 0: return float('inf')
        return TD * ( (A / denom) + B )

# =========================================================
# 🖥️ COMPONENTES UI
# =========================================================
class ModernPlot(pg.PlotWidget):
    def __init__(self, title, xl="", yl="", aspect=False):
        super().__init__()
        self.setTitle(title, color=THEME["text_main"], size="10pt")
        self.showGrid(x=True, y=True, alpha=0.2)
        if xl: self.setLabel('bottom', xl)
        if yl: self.setLabel('left', yl)
        if aspect: self.setAspectLocked(True)
        self.setBackground(THEME["bg_panel"])
        self.setStyleSheet(f"border: 1px solid {THEME['border']}; border-radius: 4px;")

class SidePanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("SidePanel")
        self.setFixedWidth(300)
        l = QVBoxLayout(self); l.setContentsMargins(15,20,15,20); l.setSpacing(15)
        
        lbl = QLabel("T2F Lab"); lbl.setObjectName("HeaderLabel"); lbl.setAlignment(Qt.AlignCenter)
        l.addWidget(lbl)
        
        # Dados
        g1 = QGroupBox("Dados"); v1 = QVBoxLayout()
        self.btn_load = QPushButton("Carregar .MAT")
        self.btn_ref = QPushButton("Fixar como Referência")
        self.lst = QListWidget(); self.lst.setMaximumHeight(80)
        self.cmb_v = QComboBox(); self.cmb_i = QComboBox()
        v1.addWidget(self.btn_load)
        v1.addWidget(self.btn_ref)
        v1.addWidget(self.lst)
        v1.addWidget(QLabel("Sinal Tensão:")); v1.addWidget(self.cmb_v)
        v1.addWidget(QLabel("Sinal Corrente:")); v1.addWidget(self.cmb_i)
        g1.setLayout(v1); l.addWidget(g1)
        
        # RMS Real
        g_rms = QGroupBox("RMS Real"); v_rms = QVBoxLayout()
        style = "font-size: 16px; font-weight: bold; padding: 6px; border-radius: 4px; background: #222; border: 1px solid #444;"
        self.lbl_rms_a = QLabel("Ia: 0.00 A"); self.lbl_rms_a.setStyleSheet(f"color: {THEME['A']}; {style}"); self.lbl_rms_a.setAlignment(Qt.AlignCenter)
        self.lbl_rms_b = QLabel("Ib: 0.00 A"); self.lbl_rms_b.setStyleSheet(f"color: {THEME['B']}; {style}"); self.lbl_rms_b.setAlignment(Qt.AlignCenter)
        self.lbl_rms_c = QLabel("Ic: 0.00 A"); self.lbl_rms_c.setStyleSheet(f"color: {THEME['C']}; {style}"); self.lbl_rms_c.setAlignment(Qt.AlignCenter)
        v_rms.addWidget(self.lbl_rms_a); v_rms.addWidget(self.lbl_rms_b); v_rms.addWidget(self.lbl_rms_c)
        g_rms.setLayout(v_rms); l.addWidget(g_rms)
        
        # Religador
        g2 = QGroupBox("Religador"); v2 = QFormLayout()
        self.cmb_curve = QComboBox(); self.cmb_curve.addItems(list(CURVES.keys()))
        self.sp_pu = QDoubleSpinBox(); self.sp_pu.setRange(0.1, 10000); self.sp_pu.setValue(25.0); self.sp_pu.setDecimals(1)
        self.sp_td = QDoubleSpinBox(); self.sp_td.setRange(0.01, 100); self.sp_td.setValue(0.5); self.sp_td.setSingleStep(0.05)
        
        v2.addRow("Curva:", self.cmb_curve)
        v2.addRow("Pickup (A):", self.sp_pu)
        v2.addRow("Dial (TMS):", self.sp_td)
        g2.setLayout(v2); l.addWidget(g2)
        
        # Simulação
        g3 = QGroupBox("Simulação"); v3 = QFormLayout()
        self.sp_speed = QSpinBox(); self.sp_speed.setRange(1, 100); self.sp_speed.setValue(1)
        v3.addRow("Velocidade:", self.sp_speed)
        g3.setLayout(v3); l.addWidget(g3)
        l.addStretch()

# =========================================================
# 🚀 APP PRINCIPAL
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("T2F Power Analysis - Leonardo Edition (Comparação)")
        self.resize(1600, 950)
        
        self.raw_mats = {}
        self.t, self.v, self.i = None, None, None
        self.t_ref, self.v_ref, self.i_ref = None, None, None
        self.idx = 0; self.playing = False
        
        self.setup_style()
        self.setup_ui()
        self.init_plots() 
        
        self.timer = QTimer(); self.timer.timeout.connect(self.update_frame)

    def setup_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {THEME['bg_app']}; }}
            QFrame#SidePanel {{ background: {THEME['bg_panel']}; border-right: 1px solid {THEME['border']}; }}
            QLabel {{ color: {THEME['text_main']}; }}
            QLabel#HeaderLabel {{ color: {THEME['accent']}; font-size: 18px; font-weight: bold; }}
            QGroupBox {{ border: 1px solid {THEME['border']}; border-radius: 6px; margin-top: 15px; color: {THEME['accent']}; font-weight: bold; }}
            QPushButton {{ background: {THEME['accent']}; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }}
            QComboBox, QListWidget, QDoubleSpinBox, QSpinBox {{ background: {THEME['bg_widget']}; color: {THEME['text_main']}; border: 1px solid {THEME['border']}; padding: 4px; }}
            QTabWidget::pane {{ border: 1px solid {THEME['border']}; }}
            QTabBar::tab {{ background: {THEME['bg_panel']}; color: {THEME['text_dim']}; padding: 8px 20px; }}
            QTabBar::tab:selected {{ background: {THEME['bg_app']}; color: {THEME['accent']}; border-top: 2px solid {THEME['accent']}; }}
        """)

    def setup_ui(self):
        w = QWidget(); self.setCentralWidget(w); h = QHBoxLayout(w); h.setContentsMargins(0,0,0,0)
        self.side = SidePanel(); h.addWidget(self.side)
        
        rhs = QWidget(); v = QVBoxLayout(rhs); v.setContentsMargins(15,15,15,15)
        self.tabs = QTabWidget()
        
        # --- TAB 1: ANÁLISE ---
        t1 = QWidget(); g1 = QGridLayout(t1)
        self.pl_vt = ModernPlot("Tensão", "s", "V")
        self.pl_it = ModernPlot("Corrente", "s", "A")
        self.pl_vp = ModernPlot("Fasores V", aspect=True)
        self.pl_ip = ModernPlot("Fasores I", aspect=True)
        self.pl_vb = ModernPlot("Seq V"); self.pl_ib = ModernPlot("Seq I")
        
        self.bg_v = pg.BarGraphItem(x=[1,2,3], height=[0,0,0], width=0.6, brushes=[THEME['V1'], THEME['V2'], THEME['V0']])
        self.bg_i = pg.BarGraphItem(x=[1,2,3], height=[0,0,0], width=0.6, brushes=[THEME['V1'], THEME['V2'], THEME['V0']])
        self.pl_vb.addItem(self.bg_v); self.pl_ib.addItem(self.bg_i)
        
        g1.addWidget(self.pl_vt, 0, 0); g1.addWidget(self.pl_it, 0, 1)
        g1.addWidget(self.pl_vp, 1, 0); g1.addWidget(self.pl_ip, 1, 1)
        g1.addWidget(self.pl_vb, 2, 0); g1.addWidget(self.pl_ib, 2, 1)
        g1.setRowStretch(0, 2); g1.setRowStretch(1, 2); g1.setRowStretch(2, 1)
        self.tabs.addTab(t1, "📊 Análise")
        
        # --- TAB 2: PROTEÇÃO ---
        t2 = QWidget(); g2 = QGridLayout(t2)
        
        self.pl_tcc = ModernPlot("Religador TCC (Corrente)", "I (A)", "t (s)")
        self.pl_tcc.setLogMode(True, True) 
        self.pl_tcc.showGrid(True, True, alpha=0.4)
        self.pl_tcc.setXRange(-2, 5.5, padding=0)
        self.pl_tcc.setYRange(-2, 4.5, padding=0)
        
        self.pl_cla = ModernPlot("Clarke (Alpha-Beta)", aspect=True)
        g2.addWidget(self.pl_tcc, 0, 0); g2.addWidget(self.pl_cla, 1, 0)
        self.tabs.addTab(t2, "🛡️ Proteção")
        
        # --- TAB 3: COMPARAÇÃO ---
        t3 = QWidget(); g3 = QGridLayout(t3)
        
        self.pl_vt_comp = ModernPlot("Tensão - Comparação", "s", "V")
        self.pl_it_comp = ModernPlot("Corrente - Comparação", "s", "A")
        self.pl_vp_comp = ModernPlot("Fasores V - Comparação", aspect=True)
        self.pl_ip_comp = ModernPlot("Fasores I - Comparação", aspect=True)
        self.pl_vb_comp = ModernPlot("Seq V - Comparação")
        self.pl_ib_comp = ModernPlot("Seq I - Comparação")
        
        g3.addWidget(self.pl_vt_comp, 0, 0)
        g3.addWidget(self.pl_it_comp, 0, 1)
        g3.addWidget(self.pl_vp_comp, 1, 0)
        g3.addWidget(self.pl_ip_comp, 1, 1)
        g3.addWidget(self.pl_vb_comp, 2, 0)
        g3.addWidget(self.pl_ib_comp, 2, 1)
        g3.setRowStretch(0, 2)
        g3.setRowStretch(1, 2)
        g3.setRowStretch(2, 1)
        
        self.tabs.addTab(t3, "🔁 Comparação")
        
        v.addWidget(self.tabs)
        
        ctrl = QFrame(); ctrl.setFixedHeight(50); hc = QHBoxLayout(ctrl)
        self.btn_p = QPushButton("▶"); self.btn_p.setFixedWidth(40)
        self.lbl_t = QLabel("0.000 s")
        self.sli = QSlider(Qt.Horizontal)
        hc.addWidget(self.btn_p); hc.addWidget(self.lbl_t); hc.addWidget(self.sli)
        v.addWidget(ctrl); h.addWidget(rhs)
        
        # Conexões
        self.side.btn_load.clicked.connect(self.load_dialog)
        self.side.btn_ref.clicked.connect(self.set_reference)
        self.side.lst.currentRowChanged.connect(self.file_sel)
        self.side.cmb_v.currentIndexChanged.connect(self.load_sigs)
        self.side.cmb_i.currentIndexChanged.connect(self.load_sigs)
        self.btn_p.clicked.connect(self.toggle); self.sli.valueChanged.connect(self.seek)
        self.side.sp_pu.valueChanged.connect(self.upd_tcc)
        self.side.sp_td.valueChanged.connect(self.upd_tcc)
        self.side.cmb_curve.currentTextChanged.connect(self.upd_tcc)

    def create_vector_arrow(self, plot, color):
        line = pg.PlotDataItem(pen=pg.mkPen(color, width=3))
        tip = pg.ScatterPlotItem(size=12, brush=color, pen=None)
        plot.addItem(line); plot.addItem(tip)
        return line, tip

    def init_plots(self):
        # TAB 1: ANÁLISE
        self.cv_va = self.pl_vt.plot(pen=THEME['A']); self.cv_vb = self.pl_vt.plot(pen=THEME['B']); self.cv_vc = self.pl_vt.plot(pen=THEME['C'])
        self.cursor_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(THEME['cursor'], width=2))
        self.pl_vt.addItem(self.cursor_v)
        self.cv_ia = self.pl_it.plot(pen=THEME['A']); self.cv_ib = self.pl_it.plot(pen=THEME['B']); self.cv_ic = self.pl_it.plot(pen=THEME['C'])
        self.cursor_i = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(THEME['cursor'], width=2))
        self.pl_it.addItem(self.cursor_i)
        
        self.vec_va_line, self.vec_va_tip = self.create_vector_arrow(self.pl_vp, THEME['A'])
        self.vec_vb_line, self.vec_vb_tip = self.create_vector_arrow(self.pl_vp, THEME['B'])
        self.vec_vc_line, self.vec_vc_tip = self.create_vector_arrow(self.pl_vp, THEME['C'])
        self.vec_ia_line, self.vec_ia_tip = self.create_vector_arrow(self.pl_ip, THEME['A'])
        self.vec_ib_line, self.vec_ib_tip = self.create_vector_arrow(self.pl_ip, THEME['B'])
        self.vec_ic_line, self.vec_ic_tip = self.create_vector_arrow(self.pl_ip, THEME['C'])
        
        # TAB 2: PROTEÇÃO
        self.ct_main = self.pl_tcc.plot(pen=pg.mkPen(THEME['Curve'], width=2))
        self.line_pickup = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', style=Qt.DotLine, width=1))
        self.pl_tcc.addItem(self.line_pickup)
        
        self.sp_a = pg.ScatterPlotItem(size=18, brush=THEME['A'], symbol='o', pen=pg.mkPen('k', width=1))
        self.sp_a.setZValue(10)
        self.sp_b = pg.ScatterPlotItem(size=18, brush=THEME['B'], symbol='t', pen=pg.mkPen('k', width=1))
        self.sp_b.setZValue(10)
        self.sp_c = pg.ScatterPlotItem(size=18, brush=THEME['C'], symbol='s', pen=pg.mkPen('k', width=1))
        self.sp_c.setZValue(10)
        
        self.pl_tcc.addItem(self.sp_a); self.pl_tcc.addItem(self.sp_b); self.pl_tcc.addItem(self.sp_c)
        
        self.txt_a = pg.TextItem(text="Ia", anchor=(0, 1), color=THEME['A']); self.pl_tcc.addItem(self.txt_a)
        self.txt_b = pg.TextItem(text="Ib", anchor=(0, 1), color=THEME['B']); self.pl_tcc.addItem(self.txt_b)
        self.txt_c = pg.TextItem(text="Ic", anchor=(0, 1), color=THEME['C']); self.pl_tcc.addItem(self.txt_c)
        
        self.upd_tcc()
        
        self.cl_tr = self.pl_cla.plot(pen=pg.mkPen('w', width=1))
        self.sp_cl = pg.ScatterPlotItem(size=10, brush=THEME['accent'], symbol='o')
        self.pl_cla.addItem(self.sp_cl)
        
        # TAB 3: COMPARAÇÃO
        self.cv_va_comp = self.pl_vt_comp.plot(pen=pg.mkPen(THEME['A'], width=2))
        self.cv_vb_comp = self.pl_vt_comp.plot(pen=pg.mkPen(THEME['B'], width=2))
        self.cv_vc_comp = self.pl_vt_comp.plot(pen=pg.mkPen(THEME['C'], width=2))
        
        self.cv_va_ref = self.pl_vt_comp.plot(pen=pg.mkPen(THEME['A'], width=2, style=Qt.DashLine))
        self.cv_vb_ref = self.pl_vt_comp.plot(pen=pg.mkPen(THEME['B'], width=2, style=Qt.DashLine))
        self.cv_vc_ref = self.pl_vt_comp.plot(pen=pg.mkPen(THEME['C'], width=2, style=Qt.DashLine))
        
        self.cv_ia_comp = self.pl_it_comp.plot(pen=pg.mkPen(THEME['A'], width=2))
        self.cv_ib_comp = self.pl_it_comp.plot(pen=pg.mkPen(THEME['B'], width=2))
        self.cv_ic_comp = self.pl_it_comp.plot(pen=pg.mkPen(THEME['C'], width=2))
        
        self.cv_ia_ref = self.pl_it_comp.plot(pen=pg.mkPen(THEME['A'], width=2, style=Qt.DashLine))
        self.cv_ib_ref = self.pl_it_comp.plot(pen=pg.mkPen(THEME['B'], width=2, style=Qt.DashLine))
        self.cv_ic_ref = self.pl_it_comp.plot(pen=pg.mkPen(THEME['C'], width=2, style=Qt.DashLine))
        
        self.vec_va_line_comp, self.vec_va_tip_comp = self.create_vector_arrow(self.pl_vp_comp, THEME['A'])
        self.vec_vb_line_comp, self.vec_vb_tip_comp = self.create_vector_arrow(self.pl_vp_comp, THEME['B'])
        self.vec_vc_line_comp, self.vec_vc_tip_comp = self.create_vector_arrow(self.pl_vp_comp, THEME['C'])
        
        self.vec_va_line_ref, self.vec_va_tip_ref = self.create_vector_arrow(self.pl_vp_comp, THEME['V1'])
        self.vec_vb_line_ref, self.vec_vb_tip_ref = self.create_vector_arrow(self.pl_vp_comp, THEME['V2'])
        self.vec_vc_line_ref, self.vec_vc_tip_ref = self.create_vector_arrow(self.pl_vp_comp, THEME['V0'])
        
        self.vec_ia_line_comp, self.vec_ia_tip_comp = self.create_vector_arrow(self.pl_ip_comp, THEME['A'])
        self.vec_ib_line_comp, self.vec_ib_tip_comp = self.create_vector_arrow(self.pl_ip_comp, THEME['B'])
        self.vec_ic_line_comp, self.vec_ic_tip_comp = self.create_vector_arrow(self.pl_ip_comp, THEME['C'])
        
        self.vec_ia_line_ref, self.vec_ia_tip_ref = self.create_vector_arrow(self.pl_ip_comp, THEME['V1'])
        self.vec_ib_line_ref, self.vec_ib_tip_ref = self.create_vector_arrow(self.pl_ip_comp, THEME['V2'])
        self.vec_ic_line_ref, self.vec_ic_tip_ref = self.create_vector_arrow(self.pl_ip_comp, THEME['V0'])
        
        self.bg_v_comp = pg.BarGraphItem(x=[0.8,1.8,2.8], height=[0,0,0], width=0.35, brushes=[THEME['V1'], THEME['V2'], THEME['V0']])
        self.bg_v_ref = pg.BarGraphItem(x=[1.2,2.2,3.2], height=[0,0,0], width=0.35, brushes=[THEME['A'], THEME['B'], THEME['C']])
        self.pl_vb_comp.addItem(self.bg_v_comp)
        self.pl_vb_comp.addItem(self.bg_v_ref)
        
        self.bg_i_comp = pg.BarGraphItem(x=[0.8,1.8,2.8], height=[0,0,0], width=0.35, brushes=[THEME['V1'], THEME['V2'], THEME['V0']])
        self.bg_i_ref = pg.BarGraphItem(x=[1.2,2.2,3.2], height=[0,0,0], width=0.35, brushes=[THEME['A'], THEME['B'], THEME['C']])
        self.pl_ib_comp.addItem(self.bg_i_comp)
        self.pl_ib_comp.addItem(self.bg_i_ref)

    def load_dialog(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Load", "", "*.mat")
        if fs:
            self.raw_mats = {}
            for f in fs:
                try:
                    name = os.path.basename(f)
                    mat = load_matlab_file(f)
                    sigs = recursive_find_signals(mat)
                    if sigs: self.raw_mats[name] = sigs
                except: pass
            
            self.side.lst.clear(); self.side.lst.addItems(self.raw_mats.keys())
            if self.raw_mats: self.side.lst.setCurrentRow(0)

    def file_sel(self, row):
        if row < 0: return
        name = self.side.lst.item(row).text()
        sigs = self.raw_mats[name]
        keys = sorted(list(sigs.keys()))
        self.side.cmb_v.blockSignals(True); self.side.cmb_i.blockSignals(True)
        self.side.cmb_v.clear(); self.side.cmb_v.addItems(keys)
        self.side.cmb_i.clear(); self.side.cmb_i.addItems(keys)
        v_cands = [k for k in keys if 'V' in k]; i_cands = [k for k in keys if 'I' in k]
        if v_cands: self.side.cmb_v.setCurrentText(v_cands[0])
        if i_cands: self.side.cmb_i.setCurrentText(i_cands[0])
        self.side.cmb_v.blockSignals(False); self.side.cmb_i.blockSignals(False)
        self.load_sigs()

    def load_sigs(self):
        if self.side.lst.currentRow() < 0: return
        name = self.side.lst.currentItem().text()
        sigs = self.raw_mats[name]
        kv = self.side.cmb_v.currentText(); ki = self.side.cmb_i.currentText()
        if kv not in sigs or ki not in sigs: return
        tv, v = sigs[kv]; ti, i = sigs[ki]
        n = min(len(tv), len(ti), len(v), len(i))
        if n == 0: return
        self.t = tv[:n]; self.v = v[:n]; self.i = i[:n]
        self.v -= np.mean(self.v, axis=0); self.i -= np.mean(self.i, axis=0)
        self.sli.setRange(0, n-1); self.sli.setValue(0); self.update_frame()

    def set_reference(self):
        """Guarda o sinal atual como referência para comparação"""
        if self.t is None or self.v is None or self.i is None:
            return
        self.t_ref = self.t.copy()
        self.v_ref = self.v.copy()
        self.i_ref = self.i.copy()
        self.update_comparison()

    def update_vector(self, line, tip, mag, ang):
        x = mag * np.cos(np.angle(ang)); y = mag * np.sin(np.angle(ang))
        line.setData([0, x], [0, y]); tip.setData([x], [y])

    def update_frame(self):
        if self.t is None: return
        idx = self.sli.value()
        step = self.side.sp_speed.value()
        
        # TAB 1: ANÁLISE
        self.cv_va.setData(self.t, self.v[:,0]); self.cv_vb.setData(self.t, self.v[:,1]); self.cv_vc.setData(self.t, self.v[:,2])
        self.cv_ia.setData(self.t, self.i[:,0]); self.cv_ib.setData(self.t, self.i[:,1]); self.cv_ic.setData(self.t, self.i[:,2])
        now = self.t[idx]; self.cursor_v.setValue(now); self.cursor_i.setValue(now)
        
        w = 128; i0 = max(0, idx-w); i1 = min(len(self.t), idx+w)
        if i1-i0 > 16:
            tw = self.t[i0:i1]
            rms_va = true_rms(self.v[i0:i1,0]); rms_vb = true_rms(self.v[i0:i1,1]); rms_vc = true_rms(self.v[i0:i1,2])
            rms_ia = true_rms(self.i[i0:i1,0]); rms_ib = true_rms(self.i[i0:i1,1]); rms_ic = true_rms(self.i[i0:i1,2])
            Ph_Va = phasor_unit(self.v[i0:i1,0], tw); Ph_Vb = phasor_unit(self.v[i0:i1,1], tw); Ph_Vc = phasor_unit(self.v[i0:i1,2], tw)
            Ph_Ia = phasor_unit(self.i[i0:i1,0], tw); Ph_Ib = phasor_unit(self.i[i0:i1,1], tw); Ph_Ic = phasor_unit(self.i[i0:i1,2], tw)
            Full_Va = rms_va * Ph_Va; Full_Vb = rms_vb * Ph_Vb; Full_Vc = rms_vc * Ph_Vc
            Full_Ia = rms_ia * Ph_Ia; Full_Ib = rms_ib * Ph_Ib; Full_Ic = rms_ic * Ph_Ic
            V0, V1, V2 = sym_components(Full_Va, Full_Vb, Full_Vc)
            I0, I1, I2 = sym_components(Full_Ia, Full_Ib, Full_Ic)
            self.bg_v.setOpts(height=[abs(V1), abs(V2), abs(V0)]); self.bg_i.setOpts(height=[abs(I1), abs(I2), abs(I0)])
            self.side.lbl_rms_a.setText(f"Ia: {rms_ia:.2f} A")
            self.side.lbl_rms_b.setText(f"Ib: {rms_ib:.2f} A")
            self.side.lbl_rms_c.setText(f"Ic: {rms_ic:.2f} A")
            self.update_vector(self.vec_va_line, self.vec_va_tip, rms_va, Ph_Va)
            self.update_vector(self.vec_vb_line, self.vec_vb_tip, rms_vb, Ph_Vb)
            self.update_vector(self.vec_vc_line, self.vec_vc_tip, rms_vc, Ph_Vc)
            self.update_vector(self.vec_ia_line, self.vec_ia_tip, rms_ia, Ph_Ia)
            self.update_vector(self.vec_ib_line, self.vec_ib_tip, rms_ib, Ph_Ib)
            self.update_vector(self.vec_ic_line, self.vec_ic_tip, rms_ic, Ph_Ic)
            lim_v = max(rms_va, rms_vb, rms_vc, 10.0) * 1.2; self.pl_vp.setXRange(-lim_v, lim_v); self.pl_vp.setYRange(-lim_v, lim_v)
            lim_i = max(rms_ia, rms_ib, rms_ic, 1.0) * 1.2; self.pl_ip.setXRange(-lim_i, lim_i); self.pl_ip.setYRange(-lim_i, lim_i)
            
            # TAB 2: PROTEÇÃO (TCC)
            Ip = self.side.sp_pu.value(); TD = self.side.sp_td.value()
            curve = self.side.cmb_curve.currentText()
            Y_INF = 2000.0 
            
            def update_pt(I_val, sp, txt):
                x_safe = max(I_val, 0.015)
                t_calc = calculate_tcc(x_safe, Ip, TD, curve)
                if t_calc == float('inf') or t_calc > Y_INF:
                    y_safe = Y_INF
                else:
                    y_safe = t_calc
                sp.setData([x_safe], [y_safe])
                txt.setPos(math.log10(x_safe), math.log10(y_safe))
                txt.setText(f"{I_val:.1f}A")

            update_pt(rms_ia, self.sp_a, self.txt_a)
            update_pt(rms_ib, self.sp_b, self.txt_b)
            update_pt(rms_ic, self.sp_c, self.txt_c)
            
        al, be = clarke_transform(self.i[:,0], self.i[:,1], self.i[:,2])
        st = max(0, idx-200)
        self.cl_tr.setData(al[st:idx], be[st:idx])
        self.sp_cl.setData(x=np.array([al[idx]]), y=np.array([be[idx]]))
        self.lbl_t.setText(f"{self.t[idx]:.3f}s")
        
        # Atualizar comparação
        self.update_comparison()
        
        if self.playing and idx < len(self.t)-1: self.sli.setValue(idx + step)

    def update_comparison(self):
        """Atualiza a aba de comparação com sinal atual + referência"""
        if self.t is None or self.v is None or self.i is None:
            return
            
        idx = self.sli.value()
        
        # --- SINAL ATUAL (LINHAS SÓLIDAS) ---
        self.cv_va_comp.setData(self.t, self.v[:,0])
        self.cv_vb_comp.setData(self.t, self.v[:,1])
        self.cv_vc_comp.setData(self.t, self.v[:,2])
        self.cv_ia_comp.setData(self.t, self.i[:,0])
        self.cv_ib_comp.setData(self.t, self.i[:,1])
        self.cv_ic_comp.setData(self.t, self.i[:,2])
        
        w = 128
        i0 = max(0, idx-w)
        i1 = min(len(self.t), idx+w)
        
        if i1-i0 > 16:
            tw = self.t[i0:i1]
            
            rms_va = true_rms(self.v[i0:i1,0])
            rms_vb = true_rms(self.v[i0:i1,1])
            rms_vc = true_rms(self.v[i0:i1,2])
            rms_ia = true_rms(self.i[i0:i1,0])
            rms_ib = true_rms(self.i[i0:i1,1])
            rms_ic = true_rms(self.i[i0:i1,2])
            
            Ph_Va = phasor_unit(self.v[i0:i1,0], tw)
            Ph_Vb = phasor_unit(self.v[i0:i1,1], tw)
            Ph_Vc = phasor_unit(self.v[i0:i1,2], tw)
            Ph_Ia = phasor_unit(self.i[i0:i1,0], tw)
            Ph_Ib = phasor_unit(self.i[i0:i1,1], tw)
            Ph_Ic = phasor_unit(self.i[i0:i1,2], tw)
            
            Full_Va = rms_va * Ph_Va
            Full_Vb = rms_vb * Ph_Vb
            Full_Vc = rms_vc * Ph_Vc
            Full_Ia = rms_ia * Ph_Ia
            Full_Ib = rms_ib * Ph_Ib
            Full_Ic = rms_ic * Ph_Ic
            
            V0, V1, V2 = sym_components(Full_Va, Full_Vb, Full_Vc)
            I0, I1, I2 = sym_components(Full_Ia, Full_Ib, Full_Ic)
            
            self.update_vector(self.vec_va_line_comp, self.vec_va_tip_comp, rms_va, Ph_Va)
            self.update_vector(self.vec_vb_line_comp, self.vec_vb_tip_comp, rms_vb, Ph_Vb)
            self.update_vector(self.vec_vc_line_comp, self.vec_vc_tip_comp, rms_vc, Ph_Vc)
            self.update_vector(self.vec_ia_line_comp, self.vec_ia_tip_comp, rms_ia, Ph_Ia)
            self.update_vector(self.vec_ib_line_comp, self.vec_ib_tip_comp, rms_ib, Ph_Ib)
            self.update_vector(self.vec_ic_line_comp, self.vec_ic_tip_comp, rms_ic, Ph_Ic)
            
            self.bg_v_comp.setOpts(height=[abs(V1), abs(V2), abs(V0)])
            self.bg_i_comp.setOpts(height=[abs(I1), abs(I2), abs(I0)])
        
        # --- SINAL DE REFERÊNCIA (LINHAS TRACEJADAS) ---
        if self.t_ref is not None and self.v_ref is not None and self.i_ref is not None:
            n = min(len(self.t_ref), len(self.v_ref), len(self.i_ref))
            
            self.cv_va_ref.setData(self.t_ref[:n], self.v_ref[:n,0])
            self.cv_vb_ref.setData(self.t_ref[:n], self.v_ref[:n,1])
            self.cv_vc_ref.setData(self.t_ref[:n], self.v_ref[:n,2])
            self.cv_ia_ref.setData(self.t_ref[:n], self.i_ref[:n,0])
            self.cv_ib_ref.setData(self.t_ref[:n], self.i_ref[:n,1])
            self.cv_ic_ref.setData(self.t_ref[:n], self.i_ref[:n,2])
            
            idx_ref = int((idx / len(self.t)) * n) if len(self.t) > 0 else 0
            i0_ref = max(0, idx_ref-w)
            i1_ref = min(n, idx_ref+w)
            
            if i1_ref-i0_ref > 16:
                tw_ref = self.t_ref[i0_ref:i1_ref]
                
                rms_va_ref = true_rms(self.v_ref[i0_ref:i1_ref,0])
                rms_vb_ref = true_rms(self.v_ref[i0_ref:i1_ref,1])
                rms_vc_ref = true_rms(self.v_ref[i0_ref:i1_ref,2])
                rms_ia_ref = true_rms(self.i_ref[i0_ref:i1_ref,0])
                rms_ib_ref = true_rms(self.i_ref[i0_ref:i1_ref,1])
                rms_ic_ref = true_rms(self.i_ref[i0_ref:i1_ref,2])
                
                Ph_Va_ref = phasor_unit(self.v_ref[i0_ref:i1_ref,0], tw_ref)
                Ph_Vb_ref = phasor_unit(self.v_ref[i0_ref:i1_ref,1], tw_ref)
                Ph_Vc_ref = phasor_unit(self.v_ref[i0_ref:i1_ref,2], tw_ref)
                Ph_Ia_ref = phasor_unit(self.i_ref[i0_ref:i1_ref,0], tw_ref)
                Ph_Ib_ref = phasor_unit(self.i_ref[i0_ref:i1_ref,1], tw_ref)
                Ph_Ic_ref = phasor_unit(self.i_ref[i0_ref:i1_ref,2], tw_ref)
                
                Full_Va_ref = rms_va_ref * Ph_Va_ref
                Full_Vb_ref = rms_vb_ref * Ph_Vb_ref
                Full_Vc_ref = rms_vc_ref * Ph_Vc_ref
                Full_Ia_ref = rms_ia_ref * Ph_Ia_ref
                Full_Ib_ref = rms_ib_ref * Ph_Ib_ref
                Full_Ic_ref = rms_ic_ref * Ph_Ic_ref
                
                V0_ref, V1_ref, V2_ref = sym_components(Full_Va_ref, Full_Vb_ref, Full_Vc_ref)
                I0_ref, I1_ref, I2_ref = sym_components(Full_Ia_ref, Full_Ib_ref, Full_Ic_ref)
                
                self.update_vector(self.vec_va_line_ref, self.vec_va_tip_ref, rms_va_ref, Ph_Va_ref)
                self.update_vector(self.vec_vb_line_ref, self.vec_vb_tip_ref, rms_vb_ref, Ph_Vb_ref)
                self.update_vector(self.vec_vc_line_ref, self.vec_vc_tip_ref, rms_vc_ref, Ph_Vc_ref)
                self.update_vector(self.vec_ia_line_ref, self.vec_ia_tip_ref, rms_ia_ref, Ph_Ia_ref)
                self.update_vector(self.vec_ib_line_ref, self.vec_ib_tip_ref, rms_ib_ref, Ph_Ib_ref)
                self.update_vector(self.vec_ic_line_ref, self.vec_ic_tip_ref, rms_ic_ref, Ph_Ic_ref)
                
                self.bg_v_ref.setOpts(height=[abs(V1_ref), abs(V2_ref), abs(V0_ref)])
                self.bg_i_ref.setOpts(height=[abs(I1_ref), abs(I2_ref), abs(I0_ref)])
                
                lim_v = max(rms_va, rms_vb, rms_vc, rms_va_ref, rms_vb_ref, rms_vc_ref, 10.0) * 1.2
                lim_i = max(rms_ia, rms_ib, rms_ic, rms_ia_ref, rms_ib_ref, rms_ic_ref, 1.0) * 1.2
                self.pl_vp_comp.setXRange(-lim_v, lim_v)
                self.pl_vp_comp.setYRange(-lim_v, lim_v)
                self.pl_ip_comp.setXRange(-lim_i, lim_i)
                self.pl_ip_comp.setYRange(-lim_i, lim_i)

    def upd_tcc(self):
        Ip = self.side.sp_pu.value(); TD = self.side.sp_td.value()
        curve = self.side.cmb_curve.currentText()
        if Ip > 0: self.line_pickup.setValue(math.log10(Ip))
        start_I = Ip * 1.01; end_I = 300000 
        if start_I < end_I:
            I_plot = np.logspace(np.log10(start_I), np.log10(end_I), 1000)
            t_plot = calculate_tcc(I_plot, Ip, TD, curve)
            self.ct_main.setData(I_plot, t_plot)
            self.pl_tcc.setXRange(-2, 5.5, padding=0)
            self.pl_tcc.setYRange(-2, 4.5, padding=0)
        else: self.ct_main.clear()

    def toggle(self):
        self.playing = not self.playing; self.btn_p.setText("⏸" if self.playing else "▶")
        if self.playing: self.timer.start(40)
        else: self.timer.stop()
        
    def seek(self): self.update_frame()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
