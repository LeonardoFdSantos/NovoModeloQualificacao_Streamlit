import sys
import os
import math
import numpy as np
from scipy.io import loadmat

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSlider, QLabel, QPushButton, QCheckBox, QDoubleSpinBox, QGroupBox, QFrame,
    QComboBox, QListWidget, QFileDialog, QFormLayout, QPlainTextEdit, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont
import pyqtgraph as pg

# --- Styling & Parameters ---

# Modern Dark Theme Colors
COLOR_BG = "#1e1e1e"
COLOR_PANEL = "#252526"
COLOR_TEXT = "#d4d4d4"
COLOR_ACCENT = "#007acc"
COLOR_ACCENT_HOVER = "#0098ff"
COLOR_BORDER = "#3e3e42"
COLOR_WIDGET_BG = "#333333"

# Plot Colors (Neon/Bright for dark background)
# Fases A, B, C
COLOR_PHASE_A = '#00d4ff'  # Neon Cyan
COLOR_PHASE_B = '#ff4b4b'  # Neon Red
COLOR_PHASE_C = '#5aff5a'  # Neon Green

# Sequências
COLOR_SEQ_1 = '#29b5e8'    # Positiva (Blue-ish)
COLOR_SEQ_2 = '#e8299c'    # Negativa (Magenta)
COLOR_SEQ_0 = '#fcc203'    # Zero (Gold)

# Trajetórias e Vetores
COLOR_TRAJ = '#ffffff'
COLOR_GRID = (255, 255, 255, 30)

# Configure PyQtGraph global look
pg.setConfigOption('background', COLOR_BG)
pg.setConfigOption('foreground', COLOR_TEXT)
pg.setConfigOptions(antialias=True)


# =========================================================
# Math Core (Preservado do seu código original)
# =========================================================
def clarke_transform(a, b, c, mode="power"):
    k = (2/3) if mode == "amp" else math.sqrt(2/3)
    alpha = k * (a - 0.5*b - 0.5*c)
    beta  = k * ((math.sqrt(3)/2)*b - (math.sqrt(3)/2)*c)
    return alpha, beta

def symmetrical_components(Va, Vb, Vc):
    a = np.exp(1j * 2*np.pi/3)
    T = (1/3) * np.array([
        [1,   1,   1],
        [1, a**2,  a],
        [1,   a, a**2]
    ], dtype=complex)
    V0, V1, V2 = T @ np.array([Va, Vb, Vc], dtype=complex)
    return V0, V1, V2

def inv_symmetrical_components(V0, V1, V2):
    a = np.exp(1j * 2*np.pi/3)
    Ti = np.array([
        [1,   1,   1],
        [1,   a, a**2],
        [1, a**2,  a]
    ], dtype=complex)
    Va, Vb, Vc = Ti @ np.array([V0, V1, V2], dtype=complex)
    return Va, Vb, Vc

def phasor_window_rms(x, t, f0=60.0, win="hann", remove_mean=True):
    x = np.asarray(x).squeeze()
    t = np.asarray(t).squeeze()
    N = len(x)
    if N < 16: return 0.0 + 0.0j
    if remove_mean: x = x - np.mean(x)
    w = np.hanning(N) if win == "hann" else np.ones(N)
    xw = x * w
    tt = t - t[0]
    exp_term = np.exp(-1j * 2*np.pi*f0*tt)
    X = np.sum(xw * exp_term)
    W = np.sum(w) + 1e-12
    return (2.0 * X / W) / np.sqrt(2)

def synth_from_phasor(Vrms, t, f0):
    return np.sqrt(2) * np.real(Vrms * np.exp(1j * 2*np.pi*f0*t))

# --- Utils de leitura MAT ---
def _unwrap_scalar(x):
    if isinstance(x, np.ndarray):
        x = x.squeeze()
        if x.dtype == object and x.size == 1: return x.item()
        if x.size == 1:
            try: return x.item()
            except: return x
    return x

def _mat_getfield(obj, name):
    if obj is None: return None
    if hasattr(obj, name): return getattr(obj, name)
    try:
        if hasattr(obj, "dtype") and obj.dtype.names and name in obj.dtype.names:
            return obj[name]
    except: pass
    return None

def extract_timeseries(mat, point_name):
    if 'ts' not in mat: return None, None
    ts_root = _unwrap_scalar(mat['ts'])
    key = f"ts_{point_name}"
    entry = ts_root.get(key, None) if isinstance(ts_root, dict) else _mat_getfield(ts_root, key)
    entry = _unwrap_scalar(entry)
    if entry is None: return None, None
    time_ = _unwrap_scalar(_mat_getfield(entry, "Time"))
    data_ = _unwrap_scalar(_mat_getfield(entry, "Data"))
    if time_ is None or data_ is None: return None, None
    t = np.asarray(time_).squeeze()
    x = np.asarray(data_)
    if x.ndim == 1: x = x.reshape(-1, 1)
    if x.ndim == 2 and x.shape[0] == 3 and x.shape[1] == t.shape[0]: x = x.T
    return t, x

def list_points(mat):
    if 'ts' not in mat: return []
    ts_root = _unwrap_scalar(mat['ts'])
    keys = []
    if isinstance(ts_root, dict): keys = list(ts_root.keys())
    elif hasattr(ts_root, "_fieldnames"): keys = list(ts_root._fieldnames)
    elif hasattr(ts_root, "dtype") and getattr(ts_root.dtype, "names", None): keys = list(ts_root.dtype.names)
    else: keys = [k for k in dir(ts_root) if (k.startswith("ts_"))]
    return sorted([str(k).replace("ts_", "") for k in keys if str(k).startswith("ts_")])


# =========================================================
# Main Application Class
# =========================================================
class T2FAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("T2F Power Analysis - Modern GUI")
        self.resize(1400, 900)
        
        # State Data
        self.mats = {}
        self.current_file = None
        self.t = None
        self.v = None
        self.i = None
        self.fs = 0.0
        self.idx = 0
        self.is_playing = False

        self.apply_stylesheet()
        self.init_ui()
        
        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.advance_frame)

    def apply_stylesheet(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {COLOR_BG};
                color: {COLOR_TEXT};
                font-family: 'Segoe UI', sans-serif;
                font-size: 10pt;
            }}
            QFrame#Sidebar {{
                background-color: {COLOR_PANEL};
                border-right: 1px solid {COLOR_BORDER};
            }}
            QLabel#SidebarTitle {{
                font-size: 14pt;
                font-weight: bold;
                color: {COLOR_ACCENT};
                margin-bottom: 10px;
            }}
            QGroupBox {{
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
                background-color: {COLOR_PANEL};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                color: {COLOR_ACCENT};
            }}
            QPushButton {{
                background-color: {COLOR_ACCENT};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLOR_ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background-color: #005c99;
            }}
            QListWidget, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
                background-color: {COLOR_WIDGET_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 4px;
                color: {COLOR_TEXT};
                padding: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {COLOR_ACCENT};
                color: white;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {COLOR_BORDER};
                height: 6px;
                background: {COLOR_WIDGET_BG};
                margin: 2px 0;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLOR_ACCENT};
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
        """)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- LEFT SIDEBAR ---
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        sidebar_layout.setSpacing(10)

        # Title
        title_label = QLabel("Controls")
        title_label.setObjectName("SidebarTitle")
        title_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title_label)

        # 1. File Loader Group
        group_files = QGroupBox("Data Input")
        layout_files = QVBoxLayout()
        
        self.btn_load = QPushButton("📂 Load .mat Files")
        self.btn_load.clicked.connect(self.load_files_dialog)
        layout_files.addWidget(self.btn_load)
        
        self.list_files = QListWidget()
        self.list_files.setMaximumHeight(100)
        self.list_files.currentRowChanged.connect(self.on_file_selected)
        layout_files.addWidget(self.list_files)
        
        form_channels = QFormLayout()
        self.cmb_v = QComboBox()
        self.cmb_i = QComboBox()
        self.cmb_v.currentIndexChanged.connect(self.reload_signals)
        self.cmb_i.currentIndexChanged.connect(self.reload_signals)
        form_channels.addRow("Voltage (V):", self.cmb_v)
        form_channels.addRow("Current (I):", self.cmb_i)
        layout_files.addLayout(form_channels)
        
        group_files.setLayout(layout_files)
        sidebar_layout.addWidget(group_files)

        # 2. Parameters Group
        group_params = QGroupBox("Processing")
        layout_params = QFormLayout()
        
        self.chk_remove_dc = QCheckBox("Remove DC Offset")
        self.chk_remove_dc.setChecked(True)
        self.chk_remove_dc.stateChanged.connect(self.recompute_all)
        
        self.cmb_clarke = QComboBox()
        self.cmb_clarke.addItems(["Power Invariant (√2/3)", "Amplitude Invariant (2/3)"])
        self.cmb_clarke.currentIndexChanged.connect(self.recompute_all)
        
        self.spin_f0 = QDoubleSpinBox()
        self.spin_f0.setValue(60.0)
        self.spin_f0.setRange(1.0, 1000.0)
        self.spin_f0.valueChanged.connect(self.recompute_all)
        
        self.spin_cycles = QSpinBox()
        self.spin_cycles.setValue(2)
        self.spin_cycles.setRange(1, 20)
        self.spin_cycles.valueChanged.connect(self.recompute_all)
        
        layout_params.addRow(self.chk_remove_dc)
        layout_params.addRow("Transform:", self.cmb_clarke)
        layout_params.addRow("Freq (Hz):", self.spin_f0)
        layout_params.addRow("Window (cyc):", self.spin_cycles)
        group_params.setLayout(layout_params)
        sidebar_layout.addWidget(group_params)

        # 3. Playback Group
        group_play = QGroupBox("Playback")
        layout_play = QVBoxLayout()
        
        self.lbl_time = QLabel("Time: 0.000 s")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        
        hbox_btns = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_play)
        hbox_btns.addWidget(self.btn_play)
        hbox_btns.addWidget(self.btn_reset)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_changed)
        
        form_speed = QFormLayout()
        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(10, 1000)
        self.spin_speed.setValue(50)
        self.spin_speed.setSuffix(" ms")
        self.spin_speed.valueChanged.connect(self.update_timer_interval)
        
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 100)
        self.spin_step.setValue(5)
        
        form_speed.addRow("Speed:", self.spin_speed)
        form_speed.addRow("Step size:", self.spin_step)
        
        layout_play.addWidget(self.lbl_time)
        layout_play.addWidget(self.slider)
        layout_play.addLayout(hbox_btns)
        layout_play.addLayout(form_speed)
        group_play.setLayout(layout_play)
        sidebar_layout.addWidget(group_play)

        # Log Area
        sidebar_layout.addWidget(QLabel("<b>Status Log:</b>"))
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        sidebar_layout.addWidget(self.txt_log)

        main_layout.addWidget(sidebar)

        # --- RIGHT CONTENT (PLOTS) ---
        content_widget = QWidget()
        grid_layout = QGridLayout(content_widget)
        grid_layout.setContentsMargins(10, 10, 10, 10)
        grid_layout.setSpacing(10)

        # Create Plot Widgets
        self.plot_v_time = self.create_plot("Voltage Waveforms (ABC)", "Time (s)", "V")
        self.plot_v_phasor = self.create_plot("Voltage Phasors (V0, V1, V2)", "Re", "Im", aspect=True)
        self.plot_v_clarke = self.create_plot("Voltage Clarke (αβ)", "α", "β", aspect=True)
        self.plot_v_seq = self.create_plot("Voltage Seq (Time Domain)", "Time (s)", "V")

        self.plot_i_time = self.create_plot("Current Waveforms (ABC)", "Time (s)", "A")
        self.plot_i_phasor = self.create_plot("Current Phasors (I0, I1, I2)", "Re", "Im", aspect=True)
        self.plot_i_clarke = self.create_plot("Current Clarke (αβ)", "α", "β", aspect=True)
        self.plot_i_seq = self.create_plot("Current Seq (Time Domain)", "Time (s)", "A")

        # Layout 4x2 Grid
        # Row 0: Voltage Time | Voltage Phasor
        grid_layout.addWidget(self.plot_v_time, 0, 0)
        grid_layout.addWidget(self.plot_v_phasor, 0, 1)
        # Row 1: Voltage Clarke | Voltage Seq Time
        grid_layout.addWidget(self.plot_v_clarke, 1, 0)
        grid_layout.addWidget(self.plot_v_seq, 1, 1)
        # Row 2: Current Time | Current Phasor
        grid_layout.addWidget(self.plot_i_time, 2, 0)
        grid_layout.addWidget(self.plot_i_phasor, 2, 1)
        # Row 3: Current Clarke | Current Seq Time
        grid_layout.addWidget(self.plot_i_clarke, 3, 0)
        grid_layout.addWidget(self.plot_i_seq, 3, 1)

        main_layout.addWidget(content_widget)

        # Init Plot Curves
        self.init_curves()
        self.log("Ready. Please load a .mat file.")

    def create_plot(self, title, xlabel, ylabel, aspect=False):
        p = pg.PlotWidget(title=title)
        p.showGrid(x=True, y=True, alpha=0.3)
        p.setLabel('bottom', xlabel)
        p.setLabel('left', ylabel)
        p.getPlotItem().setTitle(title, color=COLOR_TEXT, size='10pt')
        if aspect:
            p.setAspectLocked(True)
        return p

    def log(self, msg):
        self.txt_log.appendPlainText(f">> {msg}")

    # --- PLOT INITIALIZATION ---
    def init_curves(self):
        # 1. Voltage Time
        self.cv_va = self.plot_v_time.plot(pen=pg.mkPen(COLOR_PHASE_A, width=2))
        self.cv_vb = self.plot_v_time.plot(pen=pg.mkPen(COLOR_PHASE_B, width=2))
        self.cv_vc = self.plot_v_time.plot(pen=pg.mkPen(COLOR_PHASE_C, width=2))
        self.mk_va = self.plot_v_time.plot(pen=None, symbol='o', symbolBrush=COLOR_PHASE_A)
        self.mk_vb = self.plot_v_time.plot(pen=None, symbol='o', symbolBrush=COLOR_PHASE_B)
        self.mk_vc = self.plot_v_time.plot(pen=None, symbol='o', symbolBrush=COLOR_PHASE_C)

        # 2. Current Time
        self.cv_ia = self.plot_i_time.plot(pen=pg.mkPen(COLOR_PHASE_A, width=2))
        self.cv_ib = self.plot_i_time.plot(pen=pg.mkPen(COLOR_PHASE_B, width=2))
        self.cv_ic = self.plot_i_time.plot(pen=pg.mkPen(COLOR_PHASE_C, width=2))
        self.mk_ia = self.plot_i_time.plot(pen=None, symbol='o', symbolBrush=COLOR_PHASE_A)
        self.mk_ib = self.plot_i_time.plot(pen=None, symbol='o', symbolBrush=COLOR_PHASE_B)
        self.mk_ic = self.plot_i_time.plot(pen=None, symbol='o', symbolBrush=COLOR_PHASE_C)

        # 3. Voltage Clarke
        self.cv_v_clarke_traj = self.plot_v_clarke.plot(pen=pg.mkPen(COLOR_TRAJ, width=1))
        self.mk_v_clarke_pt = self.plot_v_clarke.plot(pen=None, symbol='o', symbolBrush=COLOR_ACCENT, symbolSize=10)

        # 4. Current Clarke
        self.cv_i_clarke_traj = self.plot_i_clarke.plot(pen=pg.mkPen(COLOR_TRAJ, width=1))
        self.mk_i_clarke_pt = self.plot_i_clarke.plot(pen=None, symbol='o', symbolBrush=COLOR_ACCENT, symbolSize=10)

        # 5. Voltage Phasors (V1, V2, V0)
        self.ln_v1 = self.plot_v_phasor.plot(pen=pg.mkPen(COLOR_SEQ_1, width=3))
        self.ln_v2 = self.plot_v_phasor.plot(pen=pg.mkPen(COLOR_SEQ_2, width=3))
        self.ln_v0 = self.plot_v_phasor.plot(pen=pg.mkPen(COLOR_SEQ_0, width=3))
        self.mk_v1 = self.plot_v_phasor.plot(pen=None, symbol='t1', symbolBrush=COLOR_SEQ_1, symbolSize=12)
        self.mk_v2 = self.plot_v_phasor.plot(pen=None, symbol='t1', symbolBrush=COLOR_SEQ_2, symbolSize=12)
        self.mk_v0 = self.plot_v_phasor.plot(pen=None, symbol='t1', symbolBrush=COLOR_SEQ_0, symbolSize=12)

        # 6. Current Phasors (I1, I2, I0)
        self.ln_i1 = self.plot_i_phasor.plot(pen=pg.mkPen(COLOR_SEQ_1, width=3))
        self.ln_i2 = self.plot_i_phasor.plot(pen=pg.mkPen(COLOR_SEQ_2, width=3))
        self.ln_i0 = self.plot_i_phasor.plot(pen=pg.mkPen(COLOR_SEQ_0, width=3))
        self.mk_i1 = self.plot_i_phasor.plot(pen=None, symbol='t1', symbolBrush=COLOR_SEQ_1, symbolSize=12)
        self.mk_i2 = self.plot_i_phasor.plot(pen=None, symbol='t1', symbolBrush=COLOR_SEQ_2, symbolSize=12)
        self.mk_i0 = self.plot_i_phasor.plot(pen=None, symbol='t1', symbolBrush=COLOR_SEQ_0, symbolSize=12)

        # 7. Voltage Seq Time
        self.cv_vseq1 = self.plot_v_seq.plot(pen=pg.mkPen(COLOR_SEQ_1, width=2))
        self.cv_vseq2 = self.plot_v_seq.plot(pen=pg.mkPen(COLOR_SEQ_2, width=2))
        self.cv_vseq0 = self.plot_v_seq.plot(pen=pg.mkPen(COLOR_SEQ_0, width=2))
        self.mk_vseq1 = self.plot_v_seq.plot(pen=None, symbol='o', symbolBrush=COLOR_SEQ_1)
        self.mk_vseq2 = self.plot_v_seq.plot(pen=None, symbol='o', symbolBrush=COLOR_SEQ_2)
        self.mk_vseq0 = self.plot_v_seq.plot(pen=None, symbol='o', symbolBrush=COLOR_SEQ_0)

        # 8. Current Seq Time
        self.cv_iseq1 = self.plot_i_seq.plot(pen=pg.mkPen(COLOR_SEQ_1, width=2))
        self.cv_iseq2 = self.plot_i_seq.plot(pen=pg.mkPen(COLOR_SEQ_2, width=2))
        self.cv_iseq0 = self.plot_i_seq.plot(pen=pg.mkPen(COLOR_SEQ_0, width=2))
        self.mk_iseq1 = self.plot_i_seq.plot(pen=None, symbol='o', symbolBrush=COLOR_SEQ_1)
        self.mk_iseq2 = self.plot_i_seq.plot(pen=None, symbol='o', symbolBrush=COLOR_SEQ_2)
        self.mk_iseq0 = self.plot_i_seq.plot(pen=None, symbol='o', symbolBrush=COLOR_SEQ_0)

    # --- LOGIC ---
    def load_files_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Load .mat files", "", "MAT Files (*.mat)")
        if paths:
            self.mats.clear()
            self.list_files.clear()
            for p in paths:
                try:
                    mat = loadmat(p, squeeze_me=False, struct_as_record=False)
                    name = os.path.basename(p)
                    self.mats[name] = mat
                    self.list_files.addItem(name)
                except Exception as e:
                    self.log(f"Error loading {p}: {e}")
            if self.mats:
                self.list_files.setCurrentRow(0)

    def on_file_selected(self, row):
        if row < 0: return
        name = self.list_files.item(row).text()
        self.current_file = name
        mat = self.mats[name]
        
        pts = list_points(mat)
        v_pts = [p for p in pts if p.upper().startswith("V")] or pts
        i_pts = [p for p in pts if p.upper().startswith("I")] or pts
        
        self.cmb_v.blockSignals(True)
        self.cmb_i.blockSignals(True)
        self.cmb_v.clear()
        self.cmb_i.clear()
        self.cmb_v.addItems(v_pts)
        self.cmb_i.addItems(i_pts)
        
        # Auto-select likely candidates
        if "V_800" in v_pts: self.cmb_v.setCurrentText("V_800")
        if "I_800" in i_pts: self.cmb_i.setCurrentText("I_800")
        
        self.cmb_v.blockSignals(False)
        self.cmb_i.blockSignals(False)
        self.reload_signals()

    def reload_signals(self):
        if not self.current_file: return
        mat = self.mats[self.current_file]
        vname = self.cmb_v.currentText()
        iname = self.cmb_i.currentText()
        
        tv, xv = extract_timeseries(mat, vname)
        ti, xi = extract_timeseries(mat, iname)
        
        if tv is None or ti is None:
            self.log("Error extracting timeseries.")
            return
            
        # Sync length
        N = min(len(tv), len(ti), xv.shape[0], xi.shape[0])
        self.t = tv[:N].astype(float)
        self.v = xv[:N, :3].astype(float)
        self.i = xi[:N, :3].astype(float)
        
        # Calculate Fs
        dt = np.mean(np.diff(self.t))
        self.fs = 1.0/dt if dt > 0 else 0.0
        
        self.slider.setRange(0, N-1)
        self.slider.setValue(0)
        self.idx = 0
        
        self.log(f"Loaded: {self.current_file} (N={N})")
        self.recompute_all()

    def recompute_all(self):
        # Triggered by parameter changes, essentially just needs a redraw
        if self.t is not None:
            self.redraw_frame()

    # --- PLAYBACK ---
    def toggle_play(self):
        if self.t is None: return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("Pause")
            self.timer.start(self.spin_speed.value())
        else:
            self.btn_play.setText("Play")
            self.timer.stop()

    def reset_play(self):
        self.is_playing = False
        self.timer.stop()
        self.btn_play.setText("Play")
        self.slider.setValue(0)
        
    def update_timer_interval(self, val):
        if self.is_playing:
            self.timer.setInterval(val)

    def on_slider_changed(self, val):
        self.idx = val
        self.redraw_frame()

    def advance_frame(self):
        step = self.spin_step.value()
        new_idx = self.idx + step
        if new_idx >= len(self.t):
            new_idx = 0
        self.slider.setValue(new_idx)

    # --- DRAWING ---
    def redraw_frame(self):
        if self.t is None: return
        
        # Params
        idx = self.idx
        t_cur = self.t[idx]
        self.lbl_time.setText(f"Time: {t_cur:.3f} s")
        
        # Data processing (Windowing)
        remove_dc = self.chk_remove_dc.isChecked()
        mode_clarke = "power" if "Power" in self.cmb_clarke.currentText() else "amp"
        f0 = self.spin_f0.value()
        win_cycles = self.spin_cycles.value()
        
        # Full Waveforms (Static background)
        v_plot = self.v
        i_plot = self.i
        if remove_dc:
            v_plot = v_plot - np.mean(v_plot, axis=0)
            i_plot = i_plot - np.mean(i_plot, axis=0)
            
        # Update Time Plots
        self.cv_va.setData(self.t, v_plot[:,0])
        self.cv_vb.setData(self.t, v_plot[:,1])
        self.cv_vc.setData(self.t, v_plot[:,2])
        self.mk_va.setData([t_cur], [v_plot[idx,0]])
        self.mk_vb.setData([t_cur], [v_plot[idx,1]])
        self.mk_vc.setData([t_cur], [v_plot[idx,2]])
        
        self.cv_ia.setData(self.t, i_plot[:,0])
        self.cv_ib.setData(self.t, i_plot[:,1])
        self.cv_ic.setData(self.t, i_plot[:,2])
        self.mk_ia.setData([t_cur], [i_plot[idx,0]])
        self.mk_ib.setData([t_cur], [i_plot[idx,1]])
        self.mk_ic.setData([t_cur], [i_plot[idx,2]])
        
        # Clarke Trajectory
        va_a, va_b = clarke_transform(v_plot[:,0], v_plot[:,1], v_plot[:,2], mode_clarke)
        ia_a, ia_b = clarke_transform(i_plot[:,0], i_plot[:,1], i_plot[:,2], mode_clarke)
        
        # Decimate trajectory for performance if needed
        stride = 1
        self.cv_v_clarke_traj.setData(va_a[::stride], va_b[::stride])
        self.mk_v_clarke_pt.setData([va_a[idx]], [va_b[idx]])
        
        self.cv_i_clarke_traj.setData(ia_a[::stride], ia_b[::stride])
        self.mk_i_clarke_pt.setData([ia_a[idx]], [ia_b[idx]])
        
        # Phasor Calc (Windowed)
        n_win = int(self.fs * win_cycles / f0) if f0 > 0 else 64
        i0 = max(0, idx - n_win//2)
        i1 = min(len(self.t), idx + n_win//2)
        
        # Ensure minimum window
        if (i1 - i0) < 16: 
            return

        tw = self.t[i0:i1]
        vw = v_plot[i0:i1]
        iw = i_plot[i0:i1]
        
        # RMS Phasors
        Va = phasor_window_rms(vw[:,0], tw, f0)
        Vb = phasor_window_rms(vw[:,1], tw, f0)
        Vc = phasor_window_rms(vw[:,2], tw, f0)
        V0, V1, V2 = symmetrical_components(Va, Vb, Vc)
        
        Ia = phasor_window_rms(iw[:,0], tw, f0)
        Ib = phasor_window_rms(iw[:,1], tw, f0)
        Ic = phasor_window_rms(iw[:,2], tw, f0)
        I0, I1, I2 = symmetrical_components(Ia, Ib, Ic)
        
        # Update Phasor Plots (Lines origin -> tip)
        self.ln_v1.setData([0, V1.real], [0, V1.imag])
        self.ln_v2.setData([0, V2.real], [0, V2.imag])
        self.ln_v0.setData([0, V0.real], [0, V0.imag])
        self.mk_v1.setData([V1.real], [V1.imag])
        self.mk_v2.setData([V2.real], [V2.imag])
        self.mk_v0.setData([V0.real], [V0.imag])
        
        self.ln_i1.setData([0, I1.real], [0, I1.imag])
        self.ln_i2.setData([0, I2.real], [0, I2.imag])
        self.ln_i0.setData([0, I0.real], [0, I0.imag])
        self.mk_i1.setData([I1.real], [I1.imag])
        self.mk_i2.setData([I2.real], [I2.imag])
        self.mk_i0.setData([I0.real], [I0.imag])
        
        # Auto-scale phasors to keep them visible
        max_v = max(abs(V1), abs(V2), abs(V0), 1.0) * 1.2
        self.plot_v_phasor.setXRange(-max_v, max_v)
        self.plot_v_phasor.setYRange(-max_v, max_v)
        
        max_i = max(abs(I1), abs(I2), abs(I0), 1.0) * 1.2
        self.plot_i_phasor.setXRange(-max_i, max_i)
        self.plot_i_phasor.setYRange(-max_i, max_i)
        
        # Sequence Time Domain Reconstruction (Phase A)
        # Reconstruct purely from the calculated phasor components
        # This shows what the sequences look like in time domain for the current window
        Va0, Va1, Va2 = inv_symmetrical_components(V0, 0, 0)[0], inv_symmetrical_components(0, V1, 0)[0], inv_symmetrical_components(0, 0, V2)[0]
        v_s1 = synth_from_phasor(Va1, self.t, f0)
        v_s2 = synth_from_phasor(Va2, self.t, f0)
        v_s0 = synth_from_phasor(Va0, self.t, f0)
        
        self.cv_vseq1.setData(self.t, v_s1)
        self.cv_vseq2.setData(self.t, v_s2)
        self.cv_vseq0.setData(self.t, v_s0)
        self.mk_vseq1.setData([t_cur], [v_s1[idx]])
        self.mk_vseq2.setData([t_cur], [v_s2[idx]])
        self.mk_vseq0.setData([t_cur], [v_s0[idx]])
        
        Ia0, Ia1, Ia2 = inv_symmetrical_components(I0, 0, 0)[0], inv_symmetrical_components(0, I1, 0)[0], inv_symmetrical_components(0, 0, I2)[0]
        i_s1 = synth_from_phasor(Ia1, self.t, f0)
        i_s2 = synth_from_phasor(Ia2, self.t, f0)
        i_s0 = synth_from_phasor(Ia0, self.t, f0)
        
        self.cv_iseq1.setData(self.t, i_s1)
        self.cv_iseq2.setData(self.t, i_s2)
        self.cv_iseq0.setData(self.t, i_s0)
        self.mk_iseq1.setData([t_cur], [i_s1[idx]])
        self.mk_iseq2.setData([t_cur], [i_s2[idx]])
        self.mk_iseq0.setData([t_cur], [i_s0[idx]])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set app font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = T2FAnalyzerApp()
    window.show()
    sys.exit(app.exec_())