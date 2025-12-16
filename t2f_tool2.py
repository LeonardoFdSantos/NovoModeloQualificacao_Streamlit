import sys
import os
import math
import numpy as np
from scipy.io import loadmat

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSlider, QLabel, QPushButton, QCheckBox, QDoubleSpinBox, QGroupBox, QFrame,
    QComboBox, QListWidget, QFileDialog, QFormLayout, QTabWidget, QSplitter,
    QSpinBox, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette
import pyqtgraph as pg

# =========================================================
# 🎨 ESTILOS E TEMAS (Dark Lab)
# =========================================================
THEME = {
    "bg_app": "#1e1e1e", "bg_panel": "#252526", "bg_widget": "#333333",
    "border": "#3e3e42", "text_main": "#e0e0e0", "text_dim": "#858585",
    "accent": "#007acc", "accent_hover": "#0098ff",
    # Cores ABC (Neon Style)
    "A": "#00ffff", "B": "#ff3333", "C": "#00ff00", 
    "V1": "#4facfe", "V2": "#f093fb", "V0": "#fcc203",
    "TCC_fast": "#ffff00", "TCC_slow": "#ff5555",
    "cursor": "#FFFF00" 
}

pg.setConfigOption('background', THEME["bg_panel"])
pg.setConfigOption('foreground', THEME["text_dim"])
pg.setConfigOptions(antialias=True)

# =========================================================
# 🛠️ LEITURA .MAT (CRAWLER RECURSIVO)
# =========================================================
def load_matlab_file(path):
    try: 
        return loadmat(path, squeeze_me=True, struct_as_record=False)
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

# =========================================================
# 🧮 MATEMÁTICA
# =========================================================
def clarke_transform(a, b, c):
    k = math.sqrt(2/3)
    alpha = k * (a - 0.5*b - 0.5*c)
    beta  = k * ((math.sqrt(3)/2)*b - (math.sqrt(3)/2)*c)
    return alpha, beta

def phasor_rms(x, t, f0=60.0):
    x = np.asarray(x); t = np.asarray(t)
    if len(x) < 16: return 0j
    x = x - np.mean(x)
    w = np.hanning(len(x))
    xw = x * w
    X = np.sum(xw * np.exp(-1j * 2*np.pi*f0*(t - t[0])))
    return (2.0 * X / np.sum(w)) / np.sqrt(2)

def sym_components(Va, Vb, Vc):
    a = np.exp(1j * 2*np.pi/3)
    V0 = (1/3) * (Va + Vb + Vc)
    V1 = (1/3) * (Va + a*Vb + a**2*Vc)
    V2 = (1/3) * (Va + a**2*Vb + a*Vc)
    return V0, V1, V2

def tcc_curve(I, Ip, TD, type="inv"):
    if Ip == 0: Ip = 0.001
    M = np.maximum(I/Ip, 1.01)
    if type == "fast": A, p, B = 0.0515, 0.02, 0.114
    else: A, p, B = 0.14, 0.02, 0.2
    return TD * (A / (M**p - 1)) + B

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
        
        # --- Seleção de Arquivos ---
        g1 = QGroupBox("Dados"); v1 = QVBoxLayout()
        self.btn_load = QPushButton("Carregar .MAT")
        self.lst = QListWidget(); self.lst.setMaximumHeight(80)
        self.cmb_v = QComboBox(); self.cmb_i = QComboBox()
        v1.addWidget(self.btn_load); v1.addWidget(self.lst)
        v1.addWidget(QLabel("Sinal Tensão:")); v1.addWidget(self.cmb_v)
        v1.addWidget(QLabel("Sinal Corrente:")); v1.addWidget(self.cmb_i)
        g1.setLayout(v1); l.addWidget(g1)
        
        # --- NOVO: Monitoramento RMS ---
        g_rms = QGroupBox("Correntes RMS (Tempo Real)"); v_rms = QVBoxLayout()
        v_rms.setSpacing(5)
        
        # Labels estilizadas para parecerem displays digitais
        base_style = "font-size: 14pt; font-weight: bold; border: 1px solid #333; border-radius: 4px; padding: 2px; background: #222;"
        
        self.lbl_rms_a = QLabel("Ia: 0.00 A")
        self.lbl_rms_a.setStyleSheet(f"color: {THEME['A']}; {base_style}")
        self.lbl_rms_a.setAlignment(Qt.AlignCenter)
        
        self.lbl_rms_b = QLabel("Ib: 0.00 A")
        self.lbl_rms_b.setStyleSheet(f"color: {THEME['B']}; {base_style}")
        self.lbl_rms_b.setAlignment(Qt.AlignCenter)
        
        self.lbl_rms_c = QLabel("Ic: 0.00 A")
        self.lbl_rms_c.setStyleSheet(f"color: {THEME['C']}; {base_style}")
        self.lbl_rms_c.setAlignment(Qt.AlignCenter)
        
        v_rms.addWidget(self.lbl_rms_a)
        v_rms.addWidget(self.lbl_rms_b)
        v_rms.addWidget(self.lbl_rms_c)
        g_rms.setLayout(v_rms)
        l.addWidget(g_rms)
        
        # --- Religador ---
        g2 = QGroupBox("Religador"); v2 = QFormLayout()
        self.sp_pu = QDoubleSpinBox(); self.sp_pu.setRange(0.1, 5000); self.sp_pu.setValue(5.0)
        self.sp_td = QDoubleSpinBox(); self.sp_td.setRange(0.05, 10); self.sp_td.setValue(0.5); self.sp_td.setSingleStep(0.1)
        v2.addRow("Pickup (A):", self.sp_pu); v2.addRow("Dial:", self.sp_td)
        g2.setLayout(v2); l.addWidget(g2)
        
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
        self.setWindowTitle("T2F Power Analysis (RMS Monitor)")
        self.resize(1600, 900)
        
        self.raw_mats = {}
        self.t, self.v, self.i = None, None, None
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
            QPushButton:hover {{ background: {THEME['accent_hover']}; }}
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
        
        # TAB 1: Análise
        t1 = QWidget(); g1 = QGridLayout(t1)
        self.pl_vt = ModernPlot("Tensão (Tempo)", "s", "V")
        self.pl_it = ModernPlot("Corrente (Tempo)", "s", "A")
        
        self.pl_vp = ModernPlot("Fasores Tensão (ABC)", aspect=True)
        self.pl_ip = ModernPlot("Fasores Corrente (ABC)", aspect=True)
        
        self.pl_vb = ModernPlot("Seq V (Barra)")
        self.bg_v = pg.BarGraphItem(x=[1,2,3], height=[0,0,0], width=0.6, brushes=[THEME['V1'], THEME['V2'], THEME['V0']])
        self.pl_vb.addItem(self.bg_v)
        
        self.pl_ib = ModernPlot("Seq I (Barra)")
        self.bg_i = pg.BarGraphItem(x=[1,2,3], height=[0,0,0], width=0.6, brushes=[THEME['V1'], THEME['V2'], THEME['V0']])
        self.pl_ib.addItem(self.bg_i)
        
        for p in [self.pl_vb, self.pl_ib]: p.getAxis('bottom').setTicks([[(1,'Pos'), (2,'Neg'), (3,'Zero')]])

        g1.addWidget(self.pl_vt, 0, 0); g1.addWidget(self.pl_it, 0, 1)
        g1.addWidget(self.pl_vp, 1, 0); g1.addWidget(self.pl_ip, 1, 1)
        g1.addWidget(self.pl_vb, 2, 0); g1.addWidget(self.pl_ib, 2, 1)
        self.tabs.addTab(t1, "📊 Análise")
        
        # TAB 2: Proteção
        t2 = QWidget(); g2 = QGridLayout(t2)
        self.pl_tcc = ModernPlot("Religador TCC (Corrente)", "I (A)", "t (s)")
        self.pl_tcc.setLogMode(True, True)
        self.pl_cla = ModernPlot("Clarke (Alpha-Beta) - Corrente", aspect=True)
        g2.addWidget(self.pl_tcc, 0, 0); g2.addWidget(self.pl_cla, 1, 0)
        self.tabs.addTab(t2, "🛡️ Proteção")
        
        v.addWidget(self.tabs)
        
        # Player Controls
        ctrl = QFrame(); ctrl.setFixedHeight(50); hc = QHBoxLayout(ctrl)
        self.btn_p = QPushButton("▶"); self.btn_p.setFixedWidth(40)
        self.lbl_t = QLabel("0.000 s")
        self.sli = QSlider(Qt.Horizontal)
        hc.addWidget(self.btn_p); hc.addWidget(self.lbl_t); hc.addWidget(self.sli)
        v.addWidget(ctrl); h.addWidget(rhs)
        
        # Signals
        self.side.btn_load.clicked.connect(self.load_dialog)
        self.side.lst.currentRowChanged.connect(self.file_sel)
        self.side.cmb_v.currentIndexChanged.connect(self.load_sigs)
        self.side.cmb_i.currentIndexChanged.connect(self.load_sigs)
        self.btn_p.clicked.connect(self.toggle); self.sli.valueChanged.connect(self.seek)
        self.side.sp_pu.valueChanged.connect(self.upd_tcc); self.side.sp_td.valueChanged.connect(self.upd_tcc)

    def create_vector_arrow(self, plot, color):
        line = pg.PlotDataItem(pen=pg.mkPen(color, width=3))
        tip = pg.ScatterPlotItem(size=12, brush=color, pen=None)
        plot.addItem(line)
        plot.addItem(tip)
        return line, tip

    def init_plots(self):
        # Waveforms
        self.cv_va = self.pl_vt.plot(pen=THEME['A']); self.cv_vb = self.pl_vt.plot(pen=THEME['B']); self.cv_vc = self.pl_vt.plot(pen=THEME['C'])
        self.cursor_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(THEME['cursor'], width=2))
        self.pl_vt.addItem(self.cursor_v)
        
        self.cv_ia = self.pl_it.plot(pen=THEME['A']); self.cv_ib = self.pl_it.plot(pen=THEME['B']); self.cv_ic = self.pl_it.plot(pen=THEME['C'])
        self.cursor_i = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(THEME['cursor'], width=2))
        self.pl_it.addItem(self.cursor_i)
        
        # Fasores ABC
        self.vec_va_line, self.vec_va_tip = self.create_vector_arrow(self.pl_vp, THEME['A'])
        self.vec_vb_line, self.vec_vb_tip = self.create_vector_arrow(self.pl_vp, THEME['B'])
        self.vec_vc_line, self.vec_vc_tip = self.create_vector_arrow(self.pl_vp, THEME['C'])
        
        self.vec_ia_line, self.vec_ia_tip = self.create_vector_arrow(self.pl_ip, THEME['A'])
        self.vec_ib_line, self.vec_ib_tip = self.create_vector_arrow(self.pl_ip, THEME['B'])
        self.vec_ic_line, self.vec_ic_tip = self.create_vector_arrow(self.pl_ip, THEME['C'])
        
        # TCC
        self.ct_f = self.pl_tcc.plot(pen=pg.mkPen(THEME['TCC_fast'], style=Qt.DashLine))
        self.ct_s = self.pl_tcc.plot(pen=pg.mkPen(THEME['TCC_slow'], width=2))
        self.sp_op = pg.ScatterPlotItem(size=15, brush='cyan', symbol='+')
        self.pl_tcc.addItem(self.sp_op)
        self.upd_tcc()
        
        # Clarke
        self.cl_tr = self.pl_cla.plot(pen=pg.mkPen('w', width=1))
        self.sp_cl = pg.ScatterPlotItem(size=10, brush=THEME['accent'], symbol='o')
        self.pl_cla.addItem(self.sp_cl)

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

    def update_vector(self, line, tip, val):
        line.setData([0, val.real], [0, val.imag])
        tip.setData([val.real], [val.imag])

    def update_frame(self):
        if self.t is None: return
        idx = self.sli.value()
        
        # 1. Waveforms
        self.cv_va.setData(self.t, self.v[:,0]); self.cv_vb.setData(self.t, self.v[:,1]); self.cv_vc.setData(self.t, self.v[:,2])
        self.cv_ia.setData(self.t, self.i[:,0]); self.cv_ib.setData(self.t, self.i[:,1]); self.cv_ic.setData(self.t, self.i[:,2])
        
        now = self.t[idx]
        self.cursor_v.setValue(now); self.cursor_i.setValue(now)
        
        # 2. Fasores e Componentes
        w = 128; i0 = max(0, idx-w); i1 = min(len(self.t), idx+w)
        if i1-i0 > 16:
            tw = self.t[i0:i1]
            Va = phasor_rms(self.v[i0:i1,0], tw); Vb = phasor_rms(self.v[i0:i1,1], tw); Vc = phasor_rms(self.v[i0:i1,2], tw)
            Ia = phasor_rms(self.i[i0:i1,0], tw); Ib = phasor_rms(self.i[i0:i1,1], tw); Ic = phasor_rms(self.i[i0:i1,2], tw)
            
            # --- ATUALIZAÇÃO NOVO MONITOR RMS NO SIDEBAR ---
            # Mostra o módulo do fasor (Valor RMS)
            self.side.lbl_rms_a.setText(f"Ia: {abs(Ia):.2f} A")
            self.side.lbl_rms_b.setText(f"Ib: {abs(Ib):.2f} A")
            self.side.lbl_rms_c.setText(f"Ic: {abs(Ic):.2f} A")

            # Update Plots
            self.update_vector(self.vec_va_line, self.vec_va_tip, Va)
            self.update_vector(self.vec_vb_line, self.vec_vb_tip, Vb)
            self.update_vector(self.vec_vc_line, self.vec_vc_tip, Vc)
            
            self.update_vector(self.vec_ia_line, self.vec_ia_tip, Ia)
            self.update_vector(self.vec_ib_line, self.vec_ib_tip, Ib)
            self.update_vector(self.vec_ic_line, self.vec_ic_tip, Ic)

            V0, V1, V2 = sym_components(Va, Vb, Vc); I0, I1, I2 = sym_components(Ia, Ib, Ic)
            
            lim_v = max(abs(Va), abs(Vb), abs(Vc), 1.0) * 1.2
            self.pl_vp.setXRange(-lim_v, lim_v); self.pl_vp.setYRange(-lim_v, lim_v)
            
            lim_i = max(abs(Ia), abs(Ib), abs(Ic), 1.0) * 1.2
            self.pl_ip.setXRange(-lim_i, lim_i); self.pl_ip.setYRange(-lim_i, lim_i)
            
            self.bg_v.setOpts(height=[abs(V1), abs(V2), abs(V0)])
            self.bg_i.setOpts(height=[abs(I1), abs(I2), abs(I0)])
            
            Im = max(abs(Ia), abs(Ib), abs(Ic))
            Ip = self.side.sp_pu.value(); TD = self.side.sp_td.value()
            trip = tcc_curve(Im, Ip, TD, "fast") if Im > Ip else 0.01
            self.sp_op.setData(x=np.array([Im]), y=np.array([trip]))
            
        al, be = clarke_transform(self.i[:,0], self.i[:,1], self.i[:,2])
        st = max(0, idx-200)
        self.cl_tr.setData(al[st:idx], be[st:idx])
        self.sp_cl.setData(x=np.array([al[idx]]), y=np.array([be[idx]]))
        
        self.lbl_t.setText(f"{self.t[idx]:.3f}s")
        if self.playing and idx < len(self.t)-1:
            step = self.side.sp_speed.value()
            self.sli.setValue(idx + step)

    def upd_tcc(self):
        Ip = self.side.sp_pu.value(); TD = self.side.sp_td.value()
        x = np.logspace(np.log10(Ip*0.5), np.log10(Ip*50), 100)
        self.ct_f.setData(x, tcc_curve(x, Ip, TD, "fast"))
        self.ct_s.setData(x, tcc_curve(x, Ip, TD, "slow"))

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