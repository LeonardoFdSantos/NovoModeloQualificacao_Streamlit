# ======================================================================
# CAPÍTULO 4 – IMPORTS, ESTILO E PARÂMETROS
# ======================================================================

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
from pathlib import Path
import seaborn as sns
import warnings
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


warnings.filterwarnings("ignore")

# Estilo dos gráficos (visual mais profissional para tese)
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["lines.linewidth"] = 2.5
plt.rcParams["lines.markersize"] = 8

# Paleta de cores padronizada (MRT vs T2F, fases e sequências)
CORES = {
    "MRT":   "#E74C3C",
    "T2F":   "#3498DB",
    "Fase_A": "#00CED1",
    "Fase_B": "#FF6347",
    "Fase_C": "#32CD32",
    "I0":    "#4169E1",
    "I1":    "#228B22",
    "I2":    "#8B008B",
}

# Caminho da pasta com os .mat (ajuste aqui quando trocar de cenário)
pasta_entrada = Path("C:/Users/Leonardo Felipe/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_11/Processados_HDF5/")
saida_fig = Path("Figuras_Cap4_Plotly")
saida_fig.mkdir(exist_ok=True)

arquivos = sorted(pasta_entrada.glob("*.mat"))  # ou "*_py.mat"

# Lista de barras que serão analisadas
barras = ["800", "816", "818", "820", "822"]

# dicionário mestre: dados_por_arquivo["nome.mat"]["I_822"]["raw"] -> array N x 3
dados_por_arquivo = {}
resultados = []
linhas = []
proc_por_arquivo_barra = {}   # chave: (arquivo, barra)

for arq in arquivos:
    with h5py.File(arq, "r") as f:
        # Conferência básica: precisa ter vetor de tempo t
        if "t" not in f.keys():
            continue

        nome = arq.name
        dados_por_arquivo[nome] = {}

        # Vetor de tempo
        t = np.array(f["t"]).flatten()
        # Posição da falta, se existir nos arquivos
        m1 = float(f["m1"][()]) if "m1" in f.keys() else None

        dados_por_arquivo[nome]["t"] = t
        dados_por_arquivo[nome]["m1"] = m1

        # Loop sobre barras e leitura de correntes/tensões
        for barra in barras:
            nome_i = f"I_{barra}_raw"
            nome_v = f"V_{barra}_raw"

            # Corrente da barra (se existir no .mat)
            if nome_i in f.keys():
                I = np.array(f[nome_i])
                # Garante forma N x 3 (N amostras, 3 fases)
                if I.ndim == 1:
                    I = np.column_stack([I] * 3)
                elif I.shape[0] == 3 and I.shape[1] > 3:
                    I = I.T
                dados_por_arquivo[nome][f"I_{barra}"] = I

            # Tensão da barra (se existir no .mat)
            if nome_v in f.keys():
                V = np.array(f[nome_v])
                if V.ndim == 1:
                    V = np.column_stack([V] * 3)
                elif V.shape[0] == 3 and V.shape[1] > 3:
                    V = V.T
                dados_por_arquivo[nome][f"V_{barra}"] = V

print("Arquivos carregados:", len(dados_por_arquivo))
list(dados_por_arquivo.keys())[:5]

# ======================================================================
# SEÇÃO 3 – PROCESSADOR DE UMA BARRA (RMS, CLARKE, SEQUÊNCIAS)
# ======================================================================

class BarraProcessor:
    """
    Processa tensões e correntes de uma barra específica:
    - Ajusta formato dos sinais (N x 3).
    - Calcula RMS deslizante (janela de 1 ciclo).
    - Calcula transformação de Clarke das correntes.
    - Calcula componentes de sequência (Fortescue): I0, I1, I2.
    """

    def __init__(self, t, v_raw, i_raw, freq=60):
        # Garante arrays 1D e 2D com tipo float
        t = np.asarray(t, dtype=float).flatten()
        v_raw = np.asarray(v_raw, dtype=float)
        i_raw = np.asarray(i_raw, dtype=float)

        self.t = t
        self.freq = float(freq)

        # Ajusta forma dos vetores (sempre N x 3)
        self.v_raw = self._fix_shape(v_raw, len(t)) / np.sqrt(3.0)
        self.i_raw = self._fix_shape(i_raw, len(t))

        # Corta todos com o mesmo comprimento L
        L = min(len(self.t), len(self.v_raw), len(self.i_raw))
        if L < 10:
            raise ValueError("Dados insuficientes.")
        self.t = self.t[:L]
        self.v_raw = self.v_raw[:L]
        self.i_raw = self.i_raw[:L]

        # Passo de tempo e taxa de amostragem
        dt = self.t[1] - self.t[0]
        self.dt = float(dt)
        self.fs = 1.0 / self.dt

        # Número de amostras em 1 ciclo (para RMS deslizante)
        self.samples = max(1, int(self.fs / self.freq))

        # ================== CÁLCULOS PRINCIPAIS ==================
        # RMS deslizante das três fases
        self.v_rms = self._rms(self.v_raw)
        self.i_rms = self._rms(self.i_raw)

        # Transformação de Clarke das correntes (alpha, beta)
        self.i_clarke = self._clarke(self.i_raw)

        # Componentes de sequência (Fortescue): magnitudes I0, I1, I2
        self.I0, self.I1, self.I2 = self._seq_components(self.i_raw)
        # Matriz N x 3 apenas por conveniência
        self.i_seq = np.stack([self.I0, self.I1, self.I2], axis=1)

    # ------------------------------------------------------------------
    # Funções auxiliares internas
    # ------------------------------------------------------------------
    def _fix_shape(self, mat, N):
        """
        Garante que 'mat' tenha formato N x 3.
        - Se vier como vetor 1D: copia para 3 colunas iguais.
        - Se vier como 3 x N: transpõe para N x 3.
        - Se já for N x 3: mantém.
        """
        if mat.ndim == 1:
            col = mat.flatten()
            return np.stack([col, col, col], axis=1)
        if mat.ndim == 2:
            if mat.shape[0] == 3 and mat.shape[1] > 3:
                return mat.T
            if mat.shape[1] == 3:
                return mat
        return np.zeros((N, 3))

    def _rms(self, x):
        """
        Calcula RMS deslizante em janela de 'self.samples' amostras
        para cada coluna (fase).
        """
        return np.sqrt(np.abs(
            scipy.ndimage.uniform_filter1d(x**2, self.samples, axis=0)
        ))

    def _clarke(self, abc):
        """
        Transformação de Clarke das correntes abc -> (alpha, beta).
        Útil para analisar desequilíbrio e trajetória no plano α-β.
        """
        a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
        alpha = (2*a - b - c)/3.0
        beta  = (b - c)/np.sqrt(3.0)
        return {"alpha": alpha, "beta": beta}

    def _seq_components(self, x):
        """
        Calcula componentes de sequência via Fortescue:
        - I0: sequência zero
        - I1: sequência positiva
        - I2: sequência negativa
        Retorna as magnitudes |I0|, |I1|, |I2| ao longo do tempo.
        """
        # Rotação para alinhar fasores em 60 Hz
        rot = np.exp(-1j * 2*np.pi * self.freq * self.t)
        ph = np.zeros_like(x, dtype=complex)

        for k in range(3):
            ph[:, k] = scipy.ndimage.uniform_filter1d(
                x[:, k] * rot, self.samples
            ) * np.sqrt(2.0)

        # Fator de fase a = e^{j 2π/3}
        a = np.exp(1j * 2*np.pi / 3.0)

        # Fortescue
        I0 = (ph[:,0] +      ph[:,1] +      ph[:,2]) / 3.0
        I1 = (ph[:,0] +  a * ph[:,1] + a**2*ph[:,2]) / 3.0
        I2 = (ph[:,0] + a**2*ph[:,1] +  a *ph[:,2]) / 3.0

        return np.abs(I0), np.abs(I1), np.abs(I2)


# ======================================================================
# SEÇÃO 4 – PROCESSAR TODAS AS COMBINAÇÕES (ARQUIVO, BARRA)
# ======================================================================

# Dicionário:
# proc_por_arquivo_barra[(nome_arquivo, barra)] = BarraProcessor(...)
proc_por_arquivo_barra = {}

for nome_arq, dados in dados_por_arquivo.items():
    t = dados["t"]
    for barra in barras:
        chave_I = f"I_{barra}"
        chave_V = f"V_{barra}"
        if chave_I in dados and chave_V in dados:
            proc = BarraProcessor(t, dados[chave_V], dados[chave_I])
            proc_por_arquivo_barra[(nome_arq, barra)] = proc

print("Total de combinações arquivo-barra processadas:",
      len(proc_por_arquivo_barra))

# Exemplo: barras disponíveis em um arquivo específico
nome_exemplo = "MRT__A_822_-_Falta_A_py.mat"
barras_disponiveis = [k[1] for k in proc_por_arquivo_barra.keys()
                      if k[0] == nome_exemplo]
print("Barras processadas em", nome_exemplo, ":", barras_disponiveis)


# ======================================================================
# SEÇÃO 5 – CRIAR DATAFRAME COM MÉTRICAS RESUMIDAS (df_resumo)
# ======================================================================

linhas = []

for (nome_arq, barra), p in proc_por_arquivo_barra.items():
    # Picos de RMS de corrente (por fase e máximo entre as fases)
    I_pico = np.max(p.i_rms, axis=0)
    I_pico_max = float(np.max(p.i_rms))

    # Pega m1 original desse arquivo
    m1 = dados_por_arquivo[nome_arq]["m1"]

    # Classificação do sistema (MRT vs T2F) a partir do nome
    if nome_arq.startswith("MRT"):
        tipo_sistema = "MRT"
    else:
        tipo_sistema = "T2F"

    # Classificação do tipo de falta a partir do nome
    if "Falta_ABC" in nome_arq:
        tipo_falta = "ABC"
    elif "Falta_AB" in nome_arq:
        tipo_falta = "AB"
    elif "Falta_AC" in nome_arq:
        tipo_falta = "AC"
    elif "Falta_BC" in nome_arq:
        tipo_falta = "BC"
    elif "Falta_A" in nome_arq:
        tipo_falta = "A-G"
    elif "Sem_Falta" in nome_arq or "Normal" in nome_arq:
        tipo_falta = "Normal"
    else:
        tipo_falta = "Desconhecido"

    # Adiciona uma linha de resumo para essa (arquivo, barra)
    linhas.append({
        "arquivo": nome_arq,
        "barra": barra,
        "m1": m1,
        "tipo_sistema": tipo_sistema,
        "tipo_falta": tipo_falta,
        "I_pico_A": I_pico[0],
        "I_pico_B": I_pico[1],
        "I_pico_C": I_pico[2],
        "I_pico_max": I_pico_max,
        "I0_max": float(np.max(p.I0)),
        "I1_max": float(np.max(p.I1)),
        "I2_max": float(np.max(p.I2)),
    })

df_resumo = pd.DataFrame(linhas)
df_resumo.head()


# ======================================================================
# SEÇÃO 6 – GRÁFICOS DETALHADOS (UM CASO, UMA BARRA)
# ======================================================================

# Escolhe um arquivo e uma barra representativos
nome = "Qualificacao__R_822_-_Falta_ABC_py.mat"
barra = "822"

p = proc_por_arquivo_barra[(nome, barra)]

# 6.1 – Tensão RMS na barra
plt.figure(figsize=(10, 4))
plt.plot(p.t, p.v_rms[:, 0], label="Va RMS", color=CORES["Fase_A"])
plt.plot(p.t, p.v_rms[:, 1], label="Vb RMS", color=CORES["Fase_B"])
plt.plot(p.t, p.v_rms[:, 2], label="Vc RMS", color=CORES["Fase_C"])
plt.xlabel("Tempo (s)")
plt.ylabel("Tensão RMS (V)")
plt.title(f"Tensões RMS – Barra {barra} – {nome}")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()

# 6.2 – Corrente RMS na barra
plt.figure(figsize=(10, 4))
plt.plot(p.t, p.i_rms[:, 0], label="Ia RMS", color=CORES["Fase_A"])
plt.plot(p.t, p.i_rms[:, 1], label="Ib RMS", color=CORES["Fase_B"])
plt.plot(p.t, p.i_rms[:, 2], label="Ic RMS", color=CORES["Fase_C"])
plt.xlabel("Tempo (s)")
plt.ylabel("Corrente RMS (A)")
plt.title(f"Correntes RMS – Barra {barra} – {nome}")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()

# 6.3 – Transformação de Clarke (XY)
plt.figure(figsize=(5, 5))
plt.plot(p.i_clarke["alpha"], p.i_clarke["beta"], color=CORES["MRT"])
plt.xlabel("Alpha")
plt.ylabel("Beta")
plt.title(f"Clarke XY – Barra {barra} – {nome}")
plt.grid(True); plt.axis("equal")
plt.tight_layout()
plt.show()

# 6.4 – Componentes de sequência (I0, I1, I2) no tempo
plt.figure(figsize=(10, 4))
plt.plot(p.t, p.I0, label="I0 (seq. zero)", color=CORES["I0"])
plt.plot(p.t, p.I1, label="I1 (seq. positiva)", color=CORES["I1"])
plt.plot(p.t, p.I2, label="I2 (seq. negativa)", color=CORES["I2"])
plt.xlabel("Tempo (s)")
plt.ylabel("Corrente (A)")
plt.title(f"Componentes de sequência – Barra {barra} – {nome}")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()


# ======================================================================
# SEÇÃO 7 – COMPARAÇÃO GLOBAL MRT × T2F EM FUNÇÃO DE m1 (TODAS AS BARRAS)
# ======================================================================

# 7.1 – Ipico_max × m1 para cada barra (MRT A-G vs T2F ABC)

for barra in barras:
    df_mrt = df_resumo[
        (df_resumo["tipo_sistema"] == "MRT") &
        (df_resumo["tipo_falta"] == "A-G") &
        (df_resumo["barra"] == barra)
    ]
    df_t2f = df_resumo[
        (df_resumo["tipo_sistema"] == "T2F") &
        (df_resumo["tipo_falta"] == "ABC") &
        (df_resumo["barra"] == barra)
    ]

    plt.figure(figsize=(7, 4))
    if not df_mrt.empty:
        plt.plot(df_mrt["m1"] * 100, df_mrt["I_pico_max"],
                 "o-", label="MRT A-G", color=CORES["MRT"])
    if not df_t2f.empty:
        plt.plot(df_t2f["m1"] * 100, df_t2f["I_pico_max"],
                 "s-", label="T2F ABC", color=CORES["T2F"])
    plt.xlabel("Posição da falta m1 (%)")
    plt.ylabel(f"I pico RMS – barra {barra} (A)")
    plt.title(f"Corrente de curto em função de m1 – Barra {barra}")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.show()

# 7.2 – I0, I1, I2 × m1 para cada barra (MRT vs T2F)

for barra in barras:
    df_mrt = df_resumo[
        (df_resumo["tipo_sistema"] == "MRT") &
        (df_resumo["tipo_falta"] == "A-G") &
        (df_resumo["barra"] == barra)
    ]
    df_t2f = df_resumo[
        (df_resumo["tipo_sistema"] == "T2F") &
        (df_resumo["tipo_falta"] == "ABC") &
        (df_resumo["barra"] == barra)
    ]

    plt.figure(figsize=(7, 4))
    if not df_mrt.empty:
        plt.plot(df_mrt["m1"] * 100, df_mrt["I0_max"], "o-", label="MRT I0", color=CORES["I0"])
        plt.plot(df_mrt["m1"] * 100, df_mrt["I1_max"], "s-", label="MRT I1", color=CORES["I1"])
        plt.plot(df_mrt["m1"] * 100, df_mrt["I2_max"], "^-", label="MRT I2", color=CORES["I2"])
    if not df_t2f.empty:
        plt.plot(df_t2f["m1"] * 100, df_t2f["I0_max"], "o--", label="T2F I0", color=CORES["I0"])
        plt.plot(df_t2f["m1"] * 100, df_t2f["I1_max"], "s--", label="T2F I1", color=CORES["I1"])
        plt.plot(df_t2f["m1"] * 100, df_t2f["I2_max"], "^--", label="T2F I2", color=CORES["I2"])
    plt.xlabel("Posição da falta m1 (%)")
    plt.ylabel(f"Componentes de sequência – barra {barra} (A)")
    plt.title(f"I0, I1, I2 em função de m1 – Barra {barra}")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.show()


# ======================================================================
# PLOTLY 1 – I_pico_max x m1 (todos arquivos, todas as barras)
# ======================================================================

for barra in barras:
    df_barra = df_resumo[df_resumo["barra"] == barra].copy()
    if df_barra.empty:
        continue

    # converte m1 para %
    df_barra["m1_pct"] = df_barra["m1"] * 100.0

    fig = px.scatter(
        df_barra,
        x="m1_pct",
        y="I_pico_max",
        color="tipo_sistema",           # MRT x T2F
        symbol="tipo_falta",           # A-G, ABC, AB, ...
        hover_data=["arquivo", "tipo_falta"],
        title=f"I_pico_max x m1 – Barra {barra}",
        labels={
            "m1_pct": "Posição da falta m1 (%)",
            "I_pico_max": "I pico RMS (A)",
            "tipo_sistema": "Sistema",
            "tipo_falta": "Tipo de falta",
        },
    )

    # adiciona linhas conectando pontos de cada combinação sistema+tipo_falta
    fig.update_traces(mode="lines+markers")

    fig.update_layout(
        template="plotly_white",
        legend=dict(title="Sistema / Tipo de falta"),
    )

    fig.show()


# ======================================================================
# PLOTLY 2 – I0, I1, I2 x m1 (todos arquivos, todas as barras)
# ======================================================================

for barra in barras:
    df_barra = df_resumo[df_resumo["barra"] == barra].copy()
    if df_barra.empty:
        continue

    df_barra["m1_pct"] = df_barra["m1"] * 100.0

    # reorganiza em formato "long" para plotar 3 curvas I0/I1/I2
    df_long = pd.melt(
        df_barra,
        id_vars=["arquivo", "barra", "m1_pct", "tipo_sistema", "tipo_falta"],
        value_vars=["I0_max", "I1_max", "I2_max"],
        var_name="Sequencia",
        value_name="Corrente",
    )

    # renomeia para rótulos bonitos
    df_long["Sequencia"] = df_long["Sequencia"].map(
        {"I0_max": "I0 (seq. zero)", "I1_max": "I1 (seq. positiva)", "I2_max": "I2 (seq. negativa)"}
    )

    fig = px.line(
        df_long,
        x="m1_pct",
        y="Corrente",
        color="Sequencia",          # I0, I1, I2
        line_dash="tipo_sistema",   # MRT vs T2F em traços diferentes
        hover_data=["arquivo", "tipo_falta"],
        title=f"I0, I1, I2 x m1 – Barra {barra}",
        labels={
            "m1_pct": "Posição da falta m1 (%)",
            "Corrente": "Corrente (A)",
            "Sequencia": "Componente",
            "tipo_sistema": "Sistema",
        },
    )

    fig.update_layout(
        template="plotly_white",
        legend=dict(title="Sequência / Sistema"),
    )

    fig.show()


# ======================================================================
# PLOTLY 3 – I0, I1, I2 x tempo (todos arquivos, barra escolhida)
# ======================================================================

barra_escolhida = "822"   # troque se quiser outra

# filtra só os pares dessa barra
pares = [(nome_arq, b) for (nome_arq, b) in proc_por_arquivo_barra.keys()
         if b == barra_escolhida]

for nome_arq, b in pares:
    p = proc_por_arquivo_barra[(nome_arq, b)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=p.t, y=p.I0,
        mode="lines",
        name="I0 (seq. zero)",
        line=dict(color=CORES["I0"])
    ))
    fig.add_trace(go.Scatter(
        x=p.t, y=p.I1,
        mode="lines",
        name="I1 (seq. positiva)",
        line=dict(color=CORES["I1"])
    ))
    fig.add_trace(go.Scatter(
        x=p.t, y=p.I2,
        mode="lines",
        name="I2 (seq. negativa)",
        line=dict(color=CORES["I2"])
    ))

    fig.update_layout(
        title=f"Componentes de sequência x tempo – Barra {barra_escolhida} – {nome_arq}",
        xaxis_title="Tempo (s)",
        yaxis_title="Corrente (A)",
        template="plotly_white",
    )

    fig.show()

# ======================================================================
# PLOTLY 4 – Clarke XY para todos os arquivos e barras
# ======================================================================

for (nome_arq, barra), p in proc_por_arquivo_barra.items():
    fig = px.scatter(
        x=p.i_clarke["alpha"],
        y=p.i_clarke["beta"],
        title=f"Clarke XY – Barra {barra} – {nome_arq}",
        labels={"x": "Alpha", "y": "Beta"},
    )
    fig.update_traces(mode="lines", line=dict(color=CORES["MRT"]))
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )
    fig.show()
