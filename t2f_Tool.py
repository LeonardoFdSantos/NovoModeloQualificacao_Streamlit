import sys
import os
import math
import numpy as np
from scipy.io import loadmat

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSlider, QLabel, QPushButton, QDoubleSpinBox, QGroupBox, QFrame,
    QComboBox, QListWidget, QFileDialog, QFormLayout, QTabWidget, QSpinBox,
    QAbstractItemView
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import pyqtgraph as pg

# =========================================================
# 📚 CONFIGURAÇÕES
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

THEME = {
    "bg_app": "#1e1e1e", "bg_panel": "#252526", "bg_widget": "#333333",
    "border": "#3e3e42", "text_main": "#e0e0e0", "text_dim": "#858585",
    "accent": "#007acc", "accent_hover": "#0098ff",
    "A": "#00ffff", "B": "#ff3333", "C": "#00ff00", 
    "V1": "#4facfe", "V2": "#f093fb", "V0": "#fcc203",
    "Curve": "#ffff00", "cursor": "#FFFF00", "Ref": "#ff6600"
}

pg.setConfigOption('background', THEME["bg_panel"])
pg.setConfigOption('foreground', THEME["text_dim"])
pg.setConfigOptions(antialias=True)

# =========================================================
# 🛠️ CÁLCULOS
# =========================================================
def calculate_tcc(I_val, Ip, TD, curve_name):
    if I_val is None: return 2000.0
    if curve_name not in CURVES: A, B, p = CURVES["IEC Standard Inverse"]
    else: A, B, p = CURVES[curve_name]
    
    safe_Ip = Ip if Ip > 0 else 0.001
    
    # Lógica escalar vs vetor
    if isinstance(I_val, np.ndarray):
        M = I_val / safe_Ip
        t = np.full_like(I_val, 2000.0)
        mask = M > 1.001
        if np.any(mask):
            denom = np.power(M[mask], p) - 1
            denom[denom == 0] = 1e-9
            t[mask] = TD * ( (A / denom) + B )
        return t
    else:
        M = I_val / safe_Ip
        if M <= 1.001: return 2000.0
        denom = (M**p - 1)
        if denom == 0: return 2000.0
        val = TD * ((A / denom) + B)
        return min(val, 2000.0)

# =========================================================
# 🖥️ COMPONENTES
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
        l = QVBoxLayout(self); l.setContentsMargins(15,20,15,20); l.setSpacing(10)
        
        lbl = QLabel("T2F Master"); lbl.setObjectName("HeaderLabel"); lbl.setAlignment(Qt.AlignCenter)
        l.addWidget(lbl)
        
        g1 = QGroupBox("Arquivos"); v1 = QVBoxLayout()
        self.btn_load = QPushButton("Adicionar Arquivos")
        self.btn_ref = QPushButton("Fixar Referência")
        self.lst_files = QListWidget()
        self.lst_files.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lst_files.setFixedHeight(150) 
        self.cmb_v = QComboBox(); self.cmb_i = QComboBox()
        v1.addWidget(self.btn_load); v1.addWidget(self.lst_files); v1.addWidget(self.btn_ref)
        v1.addWidget(QLabel("Tensão:")); v1.addWidget(self.cmb_v)
        v1.addWidget(QLabel("Corrente:")); v1.addWidget(self.cmb_i)
        g1.setLayout(v1); l.addWidget(g1)
        
        g_rms = QGroupBox("RMS"); v_rms = QVBoxLayout()
        style = "font-size: 15px; font-weight: bold; padding: 4px; border-radius: 4px; background: #222; border: 1px solid #444;"
        self.lbl_rms_a = QLabel("Ia: 0.00 A"); self.lbl_rms_a.setStyleSheet(f"color: {THEME['A']}; {style}"); self.lbl_rms_a.setAlignment(Qt.AlignCenter)
        self.lbl_rms_b = QLabel("Ib: 0.00 A"); self.lbl_rms_b.setStyleSheet(f"color: {THEME['B']}; {style}"); self.lbl_rms_b.setAlignment(Qt.AlignCenter)
        self.lbl_rms_c = QLabel("Ic: 0.00 A"); self.lbl_rms_c.setStyleSheet(f"color: {THEME['C']}; {style}"); self.lbl_rms_c.setAlignment(Qt.AlignCenter)
        v_rms.addWidget(self.lbl_rms_a); v_rms.addWidget(self.lbl_rms_b); v_rms.addWidget(self.lbl_rms_c)
        g_rms.setLayout(v_rms); l.addWidget(g_rms)
        
        g2 = QGroupBox("Religador"); v2 = QFormLayout()
        self.cmb_curve = QComboBox(); self.cmb_curve.addItems(list(CURVES.keys()))
        self.sp_pu = QDoubleSpinBox(); self.sp_pu.setRange(0.1, 10000); self.sp_pu.setValue(25.0); self.sp_pu.setDecimals(1)
        self.sp_td = QDoubleSpinBox(); self.sp_td.setRange(0.01, 100); self.sp_td.setValue(0.5); self.sp_td.setSingleStep(0.05)
        v2.addRow("Curva:", self.cmb_curve); v2.addRow("Pickup:", self.sp_pu); v2.addRow("Dial:", self.sp_td)
        g2.setLayout(v2); l.addWidget(g2)
        
        g3 = QGroupBox("Player"); v3 = QVBoxLayout()
        h_vel = QHBoxLayout()
        self.sli_speed = QSlider(Qt.Horizontal); self.sli_speed.setRange(1, 10); self.sli_speed.setValue(5)
        self.lbl_speed = QLabel("5x"); self.lbl_speed.setFixedWidth(25)
        h_vel.addWidget(QLabel("Vel:")); h_vel.addWidget(self.sli_speed); h_vel.addWidget(self.lbl_speed)
        h_step = QHBoxLayout()
        self.sp_speed = QSpinBox(); self.sp_speed.setRange(1, 100); self.sp_speed.setValue(1)
        h_step.addWidget(QLabel("Step:")); h_step.addWidget(self.sp_speed)
        v3.addLayout(h_vel); v3.addLayout(h_step)
        g3.setLayout(v3); l.addWidget(g3)
        l.addStretch()

# =========================================================
# 🚀 APP
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("T2F Power Suite - Final")
        self.resize(1600, 950)
        
        self.library = {}; self.current_raw = None; self.data_map = {}
        self.t = None; self.v_data = {}; self.i_data = {}
        self.t_ref = None; self.v_data_ref = {}; self.i_data_ref = {}

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
            QGroupBox {{ border: 1px solid {THEME['border']}; border-radius: 6px; margin-top: 10px; color: {THEME['accent']}; font-weight: bold; }}
            QPushButton {{ background: {THEME['accent']}; color: white; border: none; padding: 6px; border-radius: 4px; }}
            QListWidget {{ background: {THEME['bg_widget']}; color: {THEME['text_main']}; border: 1px solid {THEME['border']}; }}
            QListWidget::item:selected {{ background: {THEME['accent']}; color: white; }}
            QComboBox, QDoubleSpinBox, QSpinBox {{ background: {THEME['bg_widget']}; color: {THEME['text_main']}; border: 1px solid {THEME['border']}; }}
            QTabWidget::pane {{ border: 1px solid {THEME['border']}; }}
            QTabBar::tab {{ background: {THEME['bg_panel']}; color: {THEME['text_dim']}; padding: 8px 20px; }}
            QTabBar::tab:selected {{ background: {THEME['bg_app']}; color: {THEME['accent']}; border-top: 2px solid {THEME['accent']}; }}
        """)

    def setup_ui(self):
        w = QWidget(); self.setCentralWidget(w); h = QHBoxLayout(w); h.setContentsMargins(0,0,0,0)
        self.side = SidePanel(); h.addWidget(self.side)
        rhs = QWidget(); v = QVBoxLayout(rhs); v.setContentsMargins(10,10,10,10)
        self.tabs = QTabWidget()
        
        # TAB 1
        t1 = QWidget(); g1 = QGridLayout(t1)
        self.pl_vt = ModernPlot("Tensão", "s", "V"); self.pl_it = ModernPlot("Corrente", "s", "A")
        self.pl_vp = ModernPlot("Fasores V", aspect=True); self.pl_ip = ModernPlot("Fasores I", aspect=True)
        self.pl_vb = ModernPlot("Seq V"); self.pl_ib = ModernPlot("Seq I")
        self.bg_v = pg.BarGraphItem(x=[1,2,3], height=[0,0,0], width=0.6, brushes=[THEME['V1'], THEME['V2'], THEME['V0']]); self.pl_vb.addItem(self.bg_v)
        self.bg_i = pg.BarGraphItem(x=[1,2,3], height=[0,0,0], width=0.6, brushes=[THEME['V1'], THEME['V2'], THEME['V0']]); self.pl_ib.addItem(self.bg_i)
        g1.addWidget(self.pl_vt, 0, 0); g1.addWidget(self.pl_it, 0, 1)
        g1.addWidget(self.pl_vp, 1, 0); g1.addWidget(self.pl_ip, 1, 1)
        g1.addWidget(self.pl_vb, 2, 0); g1.addWidget(self.pl_ib, 2, 1)
        g1.setRowStretch(0, 2); g1.setRowStretch(1, 2); g1.setRowStretch(2, 1)
        self.tabs.addTab(t1, "📊 Análise")
        
        # TAB 2
        t2 = QWidget(); g2 = QGridLayout(t2)
        self.pl_tcc = ModernPlot("Religador TCC", "I (A)", "t (s)"); self.pl_tcc.setLogMode(True, True); self.pl_tcc.disableAutoRange()
        self.pl_tcc.setXRange(-2, 5.5); self.pl_tcc.setYRange(-2, 4.5)
        self.pl_cla = ModernPlot("Clarke", aspect=True)
        g2.addWidget(self.pl_tcc, 0, 0); g2.addWidget(self.pl_cla, 1, 0)
        self.tabs.addTab(t2, "🛡️ Proteção")
        
        # TAB 3
        t3 = QWidget(); g3 = QGridLayout(t3)
        self.pl_vt_comp = ModernPlot("V - Comparação"); self.pl_it_comp = ModernPlot("I - Comparação")
        self.pl_vp_comp = ModernPlot("Ph V - Comp", aspect=True); self.pl_ip_comp = ModernPlot("Ph I - Comp", aspect=True)
        self.pl_vb_comp = ModernPlot("Seq V - Comp"); self.pl_ib_comp = ModernPlot("Seq I - Comp")
        self.bg_v_comp = pg.BarGraphItem(x=[0.8,1.8,2.8], height=[0,0,0], width=0.35, brushes=[THEME['V1'], THEME['V2'], THEME['V0']])
        self.bg_v_ref = pg.BarGraphItem(x=[1.2,2.2,3.2], height=[0,0,0], width=0.35, brushes=[THEME['A'], THEME['B'], THEME['C']])
        self.bg_i_comp = pg.BarGraphItem(x=[0.8,1.8,2.8], height=[0,0,0], width=0.35, brushes=[THEME['V1'], THEME['V2'], THEME['V0']])
        self.bg_i_ref = pg.BarGraphItem(x=[1.2,2.2,3.2], height=[0,0,0], width=0.35, brushes=[THEME['A'], THEME['B'], THEME['C']])
        self.pl_vb_comp.addItem(self.bg_v_comp); self.pl_vb_comp.addItem(self.bg_v_ref)
        self.pl_ib_comp.addItem(self.bg_i_comp); self.pl_ib_comp.addItem(self.bg_i_ref)
        g3.addWidget(self.pl_vt_comp, 0, 0); g3.addWidget(self.pl_it_comp, 0, 1)
        g3.addWidget(self.pl_vp_comp, 1, 0); g3.addWidget(self.pl_ip_comp, 1, 1)
        g3.addWidget(self.pl_vb_comp, 2, 0); g3.addWidget(self.pl_ib_comp, 2, 1)
        g3.setRowStretch(0, 2); g3.setRowStretch(1, 2); g3.setRowStretch(2, 1)
        self.tabs.addTab(t3, "🔁 Comparação")
        
        # TAB 4
        t4 = QWidget(); g4 = QVBoxLayout(t4)
        self.pl_tcc_comp = ModernPlot("TCC - Comparação", "I (A)", "t (s)"); self.pl_tcc_comp.setLogMode(True, True); self.pl_tcc_comp.disableAutoRange()
        self.pl_tcc_comp.setXRange(-2, 5.5); self.pl_tcc_comp.setYRange(-2, 4.5)
        g4.addWidget(self.pl_tcc_comp)
        self.tabs.addTab(t4, "⚖️ TCC Comp")
        
        v.addWidget(self.tabs)
        ctrl = QFrame(); ctrl.setFixedHeight(50); hc = QHBoxLayout(ctrl)
        self.btn_p = QPushButton("▶"); self.btn_p.setFixedWidth(40)
        self.lbl_t = QLabel("0.000 s")
        self.sli = QSlider(Qt.Horizontal)
        hc.addWidget(self.btn_p); hc.addWidget(self.lbl_t); hc.addWidget(self.sli)
        v.addWidget(ctrl); h.addWidget(rhs)
        
        self.side.btn_load.clicked.connect(self.load_dialog)
        self.side.lst_files.itemClicked.connect(self.select_file_from_list)
        self.side.btn_ref.clicked.connect(self.set_reference)
        self.side.cmb_v.currentTextChanged.connect(self.load_sigs)
        self.side.cmb_i.currentTextChanged.connect(self.load_sigs)
        self.btn_p.clicked.connect(self.toggle); self.sli.valueChanged.connect(self.seek)
        self.side.sp_pu.valueChanged.connect(self.upd_tcc)
        self.side.sp_td.valueChanged.connect(self.upd_tcc)
        self.side.cmb_curve.currentTextChanged.connect(self.upd_tcc)
        self.side.sli_speed.valueChanged.connect(lambda: self.side.lbl_speed.setText(f"{self.side.sli_speed.value()}x"))

    def create_vector_arrow(self, plot, color):
        line = pg.PlotDataItem(pen=pg.mkPen(color, width=3))
        tip = pg.ScatterPlotItem(size=12, brush=color, pen=None)
        plot.addItem(line); plot.addItem(tip)
        return line, tip

    def init_plots(self):
        # --- TAB 1 ---
        self.cv_va = self.pl_vt.plot(pen=THEME['A']); self.cv_vb = self.pl_vt.plot(pen=THEME['B']); self.cv_vc = self.pl_vt.plot(pen=THEME['C'])
        self.cursor_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(THEME['cursor'], width=2)); self.pl_vt.addItem(self.cursor_v)
        self.cv_ia = self.pl_it.plot(pen=THEME['A']); self.cv_ib = self.pl_it.plot(pen=THEME['B']); self.cv_ic = self.pl_it.plot(pen=THEME['C'])
        self.cursor_i = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(THEME['cursor'], width=2)); self.pl_it.addItem(self.cursor_i)
        
        self.vec_va_l, self.vec_va_t = self.create_vector_arrow(self.pl_vp, THEME['A'])
        self.vec_vb_l, self.vec_vb_t = self.create_vector_arrow(self.pl_vp, THEME['B'])
        self.vec_vc_l, self.vec_vc_t = self.create_vector_arrow(self.pl_vp, THEME['C'])
        self.vec_ia_l, self.vec_ia_t = self.create_vector_arrow(self.pl_ip, THEME['A'])
        self.vec_ib_l, self.vec_ib_t = self.create_vector_arrow(self.pl_ip, THEME['B'])
        self.vec_ic_l, self.vec_ic_t = self.create_vector_arrow(self.pl_ip, THEME['C'])
        
        # --- TAB 2 (TCC) ---
        self.ct_main = self.pl_tcc.plot(pen=pg.mkPen(THEME['Curve'], width=2))
        self.sp_a = pg.ScatterPlotItem(size=18, brush=THEME['A'], symbol='o'); self.pl_tcc.addItem(self.sp_a)
        self.sp_b = pg.ScatterPlotItem(size=18, brush=THEME['B'], symbol='t'); self.pl_tcc.addItem(self.sp_b)
        self.sp_c = pg.ScatterPlotItem(size=18, brush=THEME['C'], symbol='s'); self.pl_tcc.addItem(self.sp_c)
        # TEXTOS TAB 2
        self.txt_a = pg.TextItem("Ia", anchor=(0,1), color=THEME['A']); self.pl_tcc.addItem(self.txt_a)
        self.txt_b = pg.TextItem("Ib", anchor=(0,1), color=THEME['B']); self.pl_tcc.addItem(self.txt_b)
        self.txt_c = pg.TextItem("Ic", anchor=(0,1), color=THEME['C']); self.pl_tcc.addItem(self.txt_c)
        
        self.cl_tr = self.pl_cla.plot(pen=pg.mkPen('w', width=1)); self.sp_cl = pg.ScatterPlotItem(size=10, brush=THEME['accent'], symbol='o'); self.pl_cla.addItem(self.sp_cl)
        
        # --- TAB 3 (COMP) ---
        self.cv_va_comp = self.pl_vt_comp.plot(pen=pg.mkPen(THEME['A'], width=2))
        self.cv_vb_comp = self.pl_vt_comp.plot(pen=pg.mkPen(THEME['B'], width=2))
        self.cv_vc_comp = self.pl_vt_comp.plot(pen=pg.mkPen(THEME['C'], width=2))
        self.cv_va_ref = self.pl_vt_comp.plot(pen=pg.mkPen(THEME['A'], width=2, style=Qt.DashLine))
        self.cv_vb_ref = self.pl_vt_comp.plot(pen=pg.mkPen(THEME['B'], width=2, style=Qt.DashLine))
        self.cv_vc_ref = self.pl_vt_comp.plot(pen=pg.mkPen(THEME['C'], width=2, style=Qt.DashLine))
        self.cursor_v_comp = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(THEME['cursor'], width=2)); self.pl_vt_comp.addItem(self.cursor_v_comp)

        self.cv_ia_comp = self.pl_it_comp.plot(pen=pg.mkPen(THEME['A'], width=2))
        self.cv_ib_comp = self.pl_it_comp.plot(pen=pg.mkPen(THEME['B'], width=2))
        self.cv_ic_comp = self.pl_it_comp.plot(pen=pg.mkPen(THEME['C'], width=2))
        self.cv_ia_ref = self.pl_it_comp.plot(pen=pg.mkPen(THEME['A'], width=2, style=Qt.DashLine))
        self.cv_ib_ref = self.pl_it_comp.plot(pen=pg.mkPen(THEME['B'], width=2, style=Qt.DashLine))
        self.cv_ic_ref = self.pl_it_comp.plot(pen=pg.mkPen(THEME['C'], width=2, style=Qt.DashLine))
        self.cursor_i_comp = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(THEME['cursor'], width=2)); self.pl_it_comp.addItem(self.cursor_i_comp)
        
        self.vec_va_comp_l, self.vec_va_comp_t = self.create_vector_arrow(self.pl_vp_comp, THEME['A'])
        self.vec_vb_comp_l, self.vec_vb_comp_t = self.create_vector_arrow(self.pl_vp_comp, THEME['B'])
        self.vec_vc_comp_l, self.vec_vc_comp_t = self.create_vector_arrow(self.pl_vp_comp, THEME['C'])
        self.vec_ia_comp_l, self.vec_ia_comp_t = self.create_vector_arrow(self.pl_ip_comp, THEME['A'])
        self.vec_ib_comp_l, self.vec_ib_comp_t = self.create_vector_arrow(self.pl_ip_comp, THEME['B'])
        self.vec_ic_comp_l, self.vec_ic_comp_t = self.create_vector_arrow(self.pl_ip_comp, THEME['C'])

        self.vec_va_ref_l, self.vec_va_ref_t = self.create_vector_arrow(self.pl_vp_comp, THEME['V1'])
        self.vec_vb_ref_l, self.vec_vb_ref_t = self.create_vector_arrow(self.pl_vp_comp, THEME['V2'])
        self.vec_vc_ref_l, self.vec_vc_ref_t = self.create_vector_arrow(self.pl_vp_comp, THEME['V0'])
        self.vec_ia_ref_l, self.vec_ia_ref_t = self.create_vector_arrow(self.pl_ip_comp, THEME['V1'])
        self.vec_ib_ref_l, self.vec_ib_ref_t = self.create_vector_arrow(self.pl_ip_comp, THEME['V2'])
        self.vec_ic_ref_l, self.vec_ic_ref_t = self.create_vector_arrow(self.pl_ip_comp, THEME['V0'])

        # --- TAB 4 (TCC COMP) ---
        self.ct_tcc_comp = self.pl_tcc_comp.plot(pen=pg.mkPen(THEME['Curve'], width=3))
        self.ct_tcc_ref = self.pl_tcc_comp.plot(pen=pg.mkPen(THEME['Ref'], width=3, style=Qt.DashLine))
        
        self.sp_a_comp = pg.ScatterPlotItem(size=18, brush=THEME['A'], symbol='o'); self.pl_tcc_comp.addItem(self.sp_a_comp)
        self.sp_b_comp = pg.ScatterPlotItem(size=18, brush=THEME['B'], symbol='t'); self.pl_tcc_comp.addItem(self.sp_b_comp)
        self.sp_c_comp = pg.ScatterPlotItem(size=18, brush=THEME['C'], symbol='s'); self.pl_tcc_comp.addItem(self.sp_c_comp)
        
        # TEXTOS TAB 4
        self.txt_a_comp = pg.TextItem("Ia", anchor=(0,1), color=THEME['A']); self.pl_tcc_comp.addItem(self.txt_a_comp)
        self.txt_b_comp = pg.TextItem("Ib", anchor=(0,1), color=THEME['B']); self.pl_tcc_comp.addItem(self.txt_b_comp)
        self.txt_c_comp = pg.TextItem("Ic", anchor=(0,1), color=THEME['C']); self.pl_tcc_comp.addItem(self.txt_c_comp)

        # Ref (Vazado/Tracejado)
        self.sp_a_ref = pg.ScatterPlotItem(size=18, brush=None, symbol='o', pen=pg.mkPen(THEME['A'], width=3)); self.pl_tcc_comp.addItem(self.sp_a_ref)
        self.sp_b_ref = pg.ScatterPlotItem(size=18, brush=None, symbol='t', pen=pg.mkPen(THEME['B'], width=3)); self.pl_tcc_comp.addItem(self.sp_b_ref)
        self.sp_c_ref = pg.ScatterPlotItem(size=18, brush=None, symbol='s', pen=pg.mkPen(THEME['C'], width=3)); self.pl_tcc_comp.addItem(self.sp_c_ref)

        self.upd_tcc()

    def load_dialog(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Arquivos", "", "*.mat")
        if not fs: return
        for f in fs:
            try:
                name = os.path.basename(f)
                if name in self.library: continue
                raw = loadmat(f, squeeze_me=True)
                self.library[name] = raw
                self.side.lst_files.addItem(name)
            except Exception as e: print(e)
        if self.side.lst_files.count()>0 and self.side.lst_files.currentRow()==-1:
            self.side.lst_files.setCurrentRow(0); self.select_file_from_list(self.side.lst_files.item(0))

    def select_file_from_list(self, item):
        if not item: return
        name = item.text(); raw = self.library.get(name)
        if not raw: return
        self.current_raw = raw
        self.data_map = {}; self.t = raw.get('t') if 't' in raw else raw.get('time')
        
        for key in raw.keys():
            if key.startswith('__') or key in ['t', 'time', 'm1']: continue
            base = key; tipo = 'raw'
            if key.endswith('_rms'): base = key[:-4]; tipo = 'rms'
            elif key.endswith('_phasor'): base = key[:-7]; tipo = 'phasor'
            elif key.endswith('_seq'): base = key[:-4]; tipo = 'seq'
            elif key.endswith('_clarke'): base = key[:-7]; tipo = 'clarke'
            elif key.endswith('_raw'): base = key[:-4]; tipo = 'raw'
            if base not in self.data_map: self.data_map[base] = {}
            self.data_map[base][tipo] = raw[key]

        keys = sorted(self.data_map.keys())
        self.side.cmb_v.blockSignals(True); self.side.cmb_i.blockSignals(True)
        self.side.cmb_v.clear(); self.side.cmb_i.clear()
        self.side.cmb_v.addItems(keys); self.side.cmb_i.addItems(keys)
        vc = [k for k in keys if 'V' in k]; ic = [k for k in keys if 'I' in k]
        if vc: self.side.cmb_v.setCurrentText(vc[0])
        if ic: self.side.cmb_i.setCurrentText(ic[0])
        self.side.cmb_v.blockSignals(False); self.side.cmb_i.blockSignals(False)
        self.load_sigs()

    def load_sigs(self):
        kv = self.side.cmb_v.currentText(); ki = self.side.cmb_i.currentText()
        if not kv or not ki: return
        self.v_data = self.data_map.get(kv, {}); self.i_data = self.data_map.get(ki, {})
        if self.t is not None:
            self.sli.setRange(0, len(self.t)-1)
            if not self.playing: self.sli.setValue(0)
            self.update_frame()

    def set_reference(self):
        if self.t is None: return
        self.t_ref = self.t; self.v_data_ref = self.v_data.copy(); self.i_data_ref = self.i_data.copy()
        self.update_frame()

    def update_vector(self, line, tip, mag, complex_val):
        ang = np.angle(complex_val); x = mag * np.cos(ang); y = mag * np.sin(ang)
        line.setData([0, x], [0, y]); tip.setData([x], [y])

    def update_frame(self):
        if self.t is None: return
        idx = self.sli.value()
        if idx >= len(self.t): idx = len(self.t)-1
        
        v_rms = self.v_data.get('rms'); v_ph = self.v_data.get('phasor'); v_seq = self.v_data.get('seq')
        i_rms = self.i_data.get('rms'); i_ph = self.i_data.get('phasor'); i_seq = self.i_data.get('seq')
        i_clk = self.i_data.get('clarke')
        val_i = i_rms[idx] if i_rms is not None else [0,0,0]
        val_v = v_rms[idx] if v_rms is not None else [0,0,0]
        
        self.side.lbl_rms_a.setText(f"Ia: {val_i[0]:.2f} A")
        self.side.lbl_rms_b.setText(f"Ib: {val_i[1]:.2f} A")
        self.side.lbl_rms_c.setText(f"Ic: {val_i[2]:.2f} A")
        self.lbl_t.setText(f"{self.t[idx]:.3f}s")
        
        # TAB 1
        if v_rms is not None:
            self.cv_va.setData(self.t, v_rms[:,0]); self.cv_vb.setData(self.t, v_rms[:,1]); self.cv_vc.setData(self.t, v_rms[:,2])
        if i_rms is not None:
            self.cv_ia.setData(self.t, i_rms[:,0]); self.cv_ib.setData(self.t, i_rms[:,1]); self.cv_ic.setData(self.t, i_rms[:,2])
        self.cursor_v.setValue(self.t[idx]); self.cursor_i.setValue(self.t[idx])

        if v_ph is not None:
            self.update_vector(self.vec_va_l, self.vec_va_t, val_v[0], v_ph[idx][0])
            self.update_vector(self.vec_vb_l, self.vec_vb_t, val_v[1], v_ph[idx][1])
            self.update_vector(self.vec_vc_l, self.vec_vc_t, val_v[2], v_ph[idx][2])
        if i_ph is not None:
            self.update_vector(self.vec_ia_l, self.vec_ia_t, val_i[0], i_ph[idx][0])
            self.update_vector(self.vec_ib_l, self.vec_ib_t, val_i[1], i_ph[idx][1])
            self.update_vector(self.vec_ic_l, self.vec_ic_t, val_i[2], i_ph[idx][2])
        if v_seq is not None: self.bg_v.setOpts(height=[abs(v_seq[idx][1]), abs(v_seq[idx][2]), abs(v_seq[idx][0])])
        if i_seq is not None: self.bg_i.setOpts(height=[abs(i_seq[idx][1]), abs(i_seq[idx][2]), abs(i_seq[idx][0])])

        # TAB 3 (ATUAL)
        if v_rms is not None:
            self.cv_va_comp.setData(self.t, v_rms[:,0]); self.cv_vb_comp.setData(self.t, v_rms[:,1]); self.cv_vc_comp.setData(self.t, v_rms[:,2])
        if i_rms is not None:
            self.cv_ia_comp.setData(self.t, i_rms[:,0]); self.cv_ib_comp.setData(self.t, i_rms[:,1]); self.cv_ic_comp.setData(self.t, i_rms[:,2])
        self.cursor_v_comp.setValue(self.t[idx]); self.cursor_i_comp.setValue(self.t[idx])

        if v_ph is not None:
            self.update_vector(self.vec_va_comp_l, self.vec_va_comp_t, val_v[0], v_ph[idx][0])
            self.update_vector(self.vec_vb_comp_l, self.vec_vb_comp_t, val_v[1], v_ph[idx][1])
            self.update_vector(self.vec_vc_comp_l, self.vec_vc_comp_t, val_v[2], v_ph[idx][2])
        if i_ph is not None:
            self.update_vector(self.vec_ia_comp_l, self.vec_ia_comp_t, val_i[0], i_ph[idx][0])
            self.update_vector(self.vec_ib_comp_l, self.vec_ib_comp_t, val_i[1], i_ph[idx][1])
            self.update_vector(self.vec_ic_comp_l, self.vec_ic_comp_t, val_i[2], i_ph[idx][2])
        if v_seq is not None: self.bg_v_comp.setOpts(height=[abs(v_seq[idx][1]), abs(v_seq[idx][2]), abs(v_seq[idx][0])])
        if i_seq is not None: self.bg_i_comp.setOpts(height=[abs(i_seq[idx][1]), abs(i_seq[idx][2]), abs(i_seq[idx][0])])

        # TCC & Clarke (Logic)
        Ip = self.side.sp_pu.value(); TD = self.side.sp_td.value(); curve = self.side.cmb_curve.currentText()
        def up_pt(I, sp, txt=None):
            x = max(I, 0.01); y = calculate_tcc(x, Ip, TD, curve)
            if y > 2000: y = 2000
            sp.setData([x], [y])
            if txt:
                txt.setPos(math.log10(x), math.log10(y))
                txt.setText(f"{I:.1f}A")
        
        # TCC Atual (Tab 2 e 4)
        up_pt(val_i[0], self.sp_a, self.txt_a); up_pt(val_i[1], self.sp_b, self.txt_b); up_pt(val_i[2], self.sp_c, self.txt_c)
        up_pt(val_i[0], self.sp_a_comp, self.txt_a_comp); up_pt(val_i[1], self.sp_b_comp, self.txt_b_comp); up_pt(val_i[2], self.sp_c_comp, self.txt_c_comp)

        if i_clk is not None:
            st = max(0, idx-200)
            self.cl_tr.setData(i_clk[st:idx, 0], i_clk[st:idx, 1])
            self.sp_cl.setData([i_clk[idx, 0]], [i_clk[idx, 1]])

        # TAB 3/4 (REFERÊNCIA)
        if self.t_ref is not None:
            idx_ref = min(idx, len(self.t_ref)-1)
            ref_v = self.v_data_ref.get('rms'); ref_i = self.i_data_ref.get('rms')
            ref_v_ph = self.v_data_ref.get('phasor'); ref_i_ph = self.i_data_ref.get('phasor')
            ref_v_seq = self.v_data_ref.get('seq'); ref_i_seq = self.i_data_ref.get('seq')
            val_ref_v = ref_v[idx_ref] if ref_v is not None else [0,0,0]
            val_ref_i = ref_i[idx_ref] if ref_i is not None else [0,0,0]

            if ref_v is not None:
                self.cv_va_ref.setData(self.t_ref, ref_v[:,0]); self.cv_vb_ref.setData(self.t_ref, ref_v[:,1]); self.cv_vc_ref.setData(self.t_ref, ref_v[:,2])
            if ref_i is not None:
                self.cv_ia_ref.setData(self.t_ref, ref_i[:,0]); self.cv_ib_ref.setData(self.t_ref, ref_i[:,1]); self.cv_ic_ref.setData(self.t_ref, ref_i[:,2])
            if ref_v_ph is not None:
                self.update_vector(self.vec_va_ref_l, self.vec_va_ref_t, val_ref_v[0], ref_v_ph[idx_ref][0])
                self.update_vector(self.vec_vb_ref_l, self.vec_vb_ref_t, val_ref_v[1], ref_v_ph[idx_ref][1])
                self.update_vector(self.vec_vc_ref_l, self.vec_vc_ref_t, val_ref_v[2], ref_v_ph[idx_ref][2])
            if ref_i_ph is not None:
                self.update_vector(self.vec_ia_ref_l, self.vec_ia_ref_t, val_ref_i[0], ref_i_ph[idx_ref][0])
                self.update_vector(self.vec_ib_ref_l, self.vec_ib_ref_t, val_ref_i[1], ref_i_ph[idx_ref][1])
                self.update_vector(self.vec_ic_ref_l, self.vec_ic_ref_t, val_ref_i[2], ref_i_ph[idx_ref][2])
            if ref_v_seq is not None: self.bg_v_ref.setOpts(height=[abs(ref_v_seq[idx_ref][1]), abs(ref_v_seq[idx_ref][2]), abs(ref_v_seq[idx_ref][0])])
            if ref_i_seq is not None: self.bg_i_ref.setOpts(height=[abs(ref_i_seq[idx_ref][1]), abs(ref_i_seq[idx_ref][2]), abs(ref_i_seq[idx_ref][0])])

            up_pt(val_ref_i[0], self.sp_a_ref); up_pt(val_ref_i[1], self.sp_b_ref); up_pt(val_ref_i[2], self.sp_c_ref)

        if self.playing and idx < len(self.t)-1:
            step = self.side.sp_speed.value()
            self.sli.setValue(idx + step)

    def upd_tcc(self):
        Ip = self.side.sp_pu.value(); TD = self.side.sp_td.value(); curve = self.side.cmb_curve.currentText()
        start_I = Ip * 1.01; end_I = 300000 
        if start_I < end_I:
            I_plot = np.logspace(np.log10(start_I), np.log10(end_I), 1000)
            t_plot = calculate_tcc(I_plot, Ip, TD, curve)
            self.ct_main.setData(I_plot, t_plot)
            self.ct_tcc_comp.setData(I_plot, t_plot)
            self.ct_tcc_ref.setData(I_plot, t_plot)

    def toggle(self):
        self.playing = not self.playing
        self.btn_p.setText("⏸" if self.playing else "▶")
        if self.playing:
            speed = self.side.sli_speed.value()
            self.timer.start(max(20, int(100 / speed)))
        else:
            self.timer.stop()
        
    def seek(self): self.update_frame()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())