# ======================================================================
# SCRIPT DE PROCESSAMENTO – CAPÍTULO 5 (COM ANÁLISE DE PROTEÇÃO COMPLETA)
# ======================================================================

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import csv
from math import isfinite, isnan

# ----------------------------------------------------------------------
# 1. CONFIGURAÇÕES GERAIS E ESTILO
# ----------------------------------------------------------------------

FREQUENCIA_REDE = 60.0  # Hz


def cm_to_inch(value):
    return value / 2.54


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "figure.figsize": (cm_to_inch(16), cm_to_inch(10)),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.framealpha": 1.0
})

PASTA_RAIZ_RESULTADOS = Path("Resultados_Tese_Cap5")


# ----------------------------------------------------------------------
# 2. FUNÇÕES NUMÉRICAS E PROCESSAMENTO DE SINAIS
# ----------------------------------------------------------------------

def calcular_rms_movel(sinal, tempo, freq_rede=FREQUENCIA_REDE):
    if len(tempo) < 2: return np.zeros_like(sinal)
    dt = tempo[1] - tempo[0]
    if dt <= 0: return np.zeros_like(sinal)
    fs = 1.0 / dt
    janela = int(round(fs / freq_rede))
    if janela < 1: janela = 1
    sinal_quadrado = sinal ** 2
    janela_media = np.ones(janela) / janela
    rms = np.sqrt(np.convolve(sinal_quadrado, janela_media, mode="same"))
    return rms


def calcular_fft_para_plot(sinal, tempo):
    dt = tempo[1] - tempo[0]
    n = len(sinal)
    fhat = np.fft.fft(sinal)
    freqs = np.fft.fftfreq(n, d=dt)
    mags = 2.0 * np.abs(fhat) / n
    mask = freqs >= 0
    return freqs[mask], mags[mask]


def calcular_v1_v3_global(sinal, tempo, freq_rede=FREQUENCIA_REDE):
    freqs, mags = calcular_fft_para_plot(sinal, tempo)
    idx_60 = np.argmin(np.abs(freqs - freq_rede))
    idx_180 = np.argmin(np.abs(freqs - 3 * freq_rede))
    return float(mags[idx_60]), float(mags[idx_180])


def get_imax_envelope_vetor(tempo, dados_tres_fases):
    rms_a = calcular_rms_movel(dados_tres_fases[:, 0], tempo)
    rms_b = calcular_rms_movel(dados_tres_fases[:, 1], tempo)
    rms_c = calcular_rms_movel(dados_tres_fases[:, 2], tempo)
    return np.maximum.reduce([rms_a, rms_b, rms_c])


def extrair_fasor_dinamico(sinal, tempo, freq=FREQUENCIA_REDE):
    dt = tempo[1] - tempo[0]
    if dt <= 0: return np.zeros_like(sinal, dtype=complex)
    samples_per_cycle = int((1.0 / freq) / dt)
    if samples_per_cycle < 1: samples_per_cycle = 1
    t_window = np.arange(samples_per_cycle) * dt
    kernel_cos = np.cos(2 * np.pi * freq * t_window) * (2 / samples_per_cycle)
    kernel_sin = np.sin(2 * np.pi * freq * t_window) * (2 / samples_per_cycle)
    real_part = np.convolve(sinal, kernel_cos, mode="same")
    imag_part = np.convolve(sinal, kernel_sin, mode="same")
    return real_part - 1j * imag_part


def calcular_componentes_simetricas_tempo(dados_3fases, tempo, freq=FREQUENCIA_REDE):
    Va = extrair_fasor_dinamico(dados_3fases[:, 0], tempo, freq)
    Vb = extrair_fasor_dinamico(dados_3fases[:, 1], tempo, freq)
    Vc = extrair_fasor_dinamico(dados_3fases[:, 2], tempo, freq)
    a = np.exp(1j * 2 * np.pi / 3)
    a2 = a ** 2
    V0 = (Va + Vb + Vc) / 3.0
    V1 = (Va + a * Vb + a2 * Vc) / 3.0
    V2 = (Va + a2 * Vb + a * Vc) / 3.0
    return np.abs(V0), np.abs(V1), np.abs(V2)


def clarke_power_invariant(ia, ib, ic):
    k = np.sqrt(2.0 / 3.0)
    i_alpha = k * (ia - 0.5 * ib - 0.5 * ic)
    i_beta = k * ((np.sqrt(3) / 2) * ib - (np.sqrt(3) / 2) * ic)
    return i_alpha, i_beta


# ----------------------------------------------------------------------
# 3. FUNÇÕES DE PROTEÇÃO (CURVAS TCC E LÓGICA DE COMPARAÇÃO)
# ----------------------------------------------------------------------

# Definição das constantes das curvas (k, c, alpha)
CURVAS_DEFS = {
    # IEC 60255
    "IEC_C1": (0.14, 0.0, 0.02),  # Standard Inverse
    "IEC_C2": (13.5, 0.0, 1.0),  # Very Inverse
    "IEC_C3": (80.0, 0.0, 2.0),  # Extremely Inverse
    # IEEE C37.112
    "IEEE_U1": (0.0515, 0.114, 0.02),  # Moderately Inverse
    "IEEE_U2": (19.61, 0.491, 2.0),  # Very Inverse
    "IEEE_U3": (28.2, 0.1217, 2.0)  # Extremely Inverse
}


def calcular_tempo_curva(M, TMS_TD, curva_nome):
    """Calcula tempo de operação dado o múltiplo M e a curva."""
    if M <= 1.0:
        return np.inf

    if curva_nome not in CURVAS_DEFS:
        return np.inf

    k, c, alpha = CURVAS_DEFS[curva_nome]

    # Diferenciação básica IEC vs IEEE na fórmula (apenas na constante de tempo se necessário)
    # A fórmula geral: t(M) = TD * (k / (M^alpha - 1) + c)
    # Para IEC: c costuma ser 0.

    # Obs: A fórmula IEEE padrão é t = TD * (k/(M^p -1) + c).
    # A fórmula IEC padrão é t = TMS * k / (M^alpha - 1). (c=0)
    # O dicionário acima já cobre os coeficientes.

    val = TMS_TD * ((k) / (M ** alpha - 1.0) + c)
    return max(0.0, val)  # Garantir não negativo


def plotar_curvas_tcc_religadores(caminho_salvar):
    M = np.logspace(0.01, 2, 500)
    TMS_plot = 0.1

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(10)))

    for nome, params in CURVAS_DEFS.items():
        # Usa a função genérica para plotar
        # Precisamos vetorizar ou chamar em loop
        tempos = [calcular_tempo_curva(m, TMS_plot, nome) for m in M]
        plt.plot(M, tempos, label=f"{nome}", linewidth=1.5)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Múltiplo da Corrente de Pickup ($M = I_{cc} / I_{pickup}$)")
    plt.ylabel("Tempo de Atuação (s)")
    plt.title(f"Curvas TCC (TMS/TD = {TMS_plot})")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(caminho_salvar, format="svg", facecolor="white")
    plt.close()


# ----------------------------------------------------------------------
# 4. FUNÇÕES DE PLOTAGEM (TODOS OS TIPOS)
# ----------------------------------------------------------------------

def gerar_grafico_fft_espectro(tempo, dados_raw, caminho_salvar, ylabel, limite_freq=1000):
    freqs_a, mag_a = calcular_fft_para_plot(dados_raw[:, 0], tempo)
    freqs_b, mag_b = calcular_fft_para_plot(dados_raw[:, 1], tempo)
    freqs_c, mag_c = calcular_fft_para_plot(dados_raw[:, 2], tempo)

    plt.figure()
    plt.plot(freqs_a, mag_a, color="red", label="Fase A", linewidth=1.2, alpha=0.8)
    plt.plot(freqs_b, mag_b, color="blue", label="Fase B", linewidth=1.2, alpha=0.8)
    plt.plot(freqs_c, mag_c, color="green", label="Fase C", linewidth=1.2, alpha=0.8)
    plt.xlabel("Frequência (Hz)")
    plt.ylabel(f"Amplitude {ylabel} (Pico)")
    plt.xlim(0, limite_freq)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(caminho_salvar, format="svg", facecolor="white")
    plt.close()


def plotar_barras_harmonicas_v1v3(v1, v3, nome_variavel, caminho_salvar, label_y_unit="V"):
    plt.figure(figsize=(cm_to_inch(10), cm_to_inch(10)))
    componentes = ['Fundamental\n(60 Hz)', '3ª Harmônica\n(180 Hz)']
    valores = [v1, v3]
    cores = ['darkblue', 'darkred']
    plt.bar(componentes, valores, color=cores, width=0.5)
    for i, v in enumerate(valores):
        plt.text(i, v + (max(valores) * 0.01 if max(valores) > 0 else 0), f"{v:.1f}", ha='center', fontsize=10)
    plt.ylabel(f"Magnitude de Pico ({label_y_unit})")
    plt.title(f"Análise Harmônica: {nome_variavel}")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(caminho_salvar, format="svg", facecolor="white")
    plt.close()


def plotar_envelope_rms_maximo(tempo, dados, caminho_salvar, label_y="Corrente"):
    imax_vetor = get_imax_envelope_vetor(tempo, dados)
    pico_max = np.max(imax_vetor)
    plt.figure()
    plt.plot(tempo, imax_vetor, color="black", linewidth=1.5, label="Envelope Máximo")
    plt.xlabel("Tempo (s)")
    plt.ylabel(f"{label_y} Máxima (RMS)")
    plt.title(f"Envelope RMS - {label_y}")
    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
    plt.text(0.5, 0.9, f"Max RMS: {pico_max:.2f}", transform=plt.gca().transAxes,
             fontsize=11, ha="center", bbox=props)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(caminho_salvar, format="svg", facecolor="white")
    plt.close()


def plotar_rms_tres_fases_com_zoom(tempo, dados, nome_variavel, caminho_salvar, label_y="Corrente"):
    rms_a = calcular_rms_movel(dados[:, 0], tempo)
    rms_b = calcular_rms_movel(dados[:, 1], tempo)
    rms_c = calcular_rms_movel(dados[:, 2], tempo)
    imax_vetor = np.maximum.reduce([rms_a, rms_b, rms_c])
    idx_pico = np.argmax(imax_vetor)
    t_pico = tempo[idx_pico]
    valor_pico = imax_vetor[idx_pico]

    fig = plt.figure(figsize=(cm_to_inch(18), cm_to_inch(16)))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(tempo, rms_a, color="blue", label="Fase A")
    ax_main.plot(tempo, rms_b, color="red", label="Fase B")
    ax_main.plot(tempo, rms_c, color="green", label="Fase C")
    ax_main.set_ylabel(f"{label_y} RMS")
    titulo = nome_variavel.replace("_raw", "").replace("_", " ")
    ax_main.set_title(f"RMS Três Fases - {titulo}")
    ax_main.legend(loc="upper right", fontsize=10)
    ax_main.grid(True, linestyle=":", alpha=0.5)

    t_z1_ini = max(tempo[0], t_pico - 0.20)
    t_z1_fim = max(tempo[0], t_pico - 0.05)
    t_z2_ini = max(tempo[0], t_pico - 0.05)
    t_z2_fim = min(tempo[-1], t_pico + 0.15)

    ax_z1 = fig.add_subplot(gs[1, 0])
    ax_z1.plot(tempo, rms_a, "b", tempo, rms_b, "r", tempo, rms_c, "g")
    ax_z1.set_xlim(t_z1_ini, t_z1_fim)
    mask1 = (tempo >= t_z1_ini) & (tempo <= t_z1_fim)
    if np.any(mask1):
        y_local = np.concatenate([rms_a[mask1], rms_b[mask1], rms_c[mask1]])
        ax_z1.set_ylim(np.min(y_local) * 0.9, np.max(y_local) * 1.1)
    ax_z1.set_title("Zoom: Pré-Evento")
    ax_z1.set_xlabel("Tempo (s)")
    ax_z1.grid(True, linestyle=":", alpha=0.5)

    ax_z2 = fig.add_subplot(gs[1, 1])
    ax_z2.plot(tempo, rms_a, "b", tempo, rms_b, "r", tempo, rms_c, "g")
    ax_z2.set_xlim(t_z2_ini, t_z2_fim)
    ax_z2.axvline(x=t_pico, color='k', linestyle='--', alpha=0.3)
    ax_z2.set_ylim(0, valor_pico * 1.1)
    ax_z2.set_title("Zoom: Evento Principal")
    ax_z2.set_xlabel("Tempo (s)")
    ax_z2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(caminho_salvar, format="svg", facecolor="white")
    plt.close()


def plotar_sequencias_simetricas(tempo, seq0, seq1, seq2, nome_variavel, caminho_salvar, label_y="Tensão"):
    plt.figure()
    plt.plot(tempo, seq1, color="blue", label="Positiva (1)", linewidth=1.5)
    plt.plot(tempo, seq2, color="red", label="Negativa (2)", linewidth=1.5, linestyle="--")
    plt.plot(tempo, seq0, color="green", label="Zero (0)", linewidth=1.5, linestyle=":")
    plt.xlabel("Tempo (s)")
    plt.ylabel(f"Magnitude {label_y} (RMS)")
    titulo = nome_variavel.replace("_raw", "").replace("_", " ")
    plt.title(f"Componentes Simétricas - {titulo}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(caminho_salvar, format="svg", facecolor="white")
    plt.close()


def plotar_trajetoria_clarke(tempo, dados, nome_variavel, caminho_salvar):
    ia, ib, ic = dados[:, 0], dados[:, 1], dados[:, 2]
    i_alpha, i_beta = clarke_power_invariant(ia, ib, ic)
    plt.figure(figsize=(cm_to_inch(10), cm_to_inch(10)))
    plt.plot(i_alpha, i_beta, color="black", linewidth=1.2)
    titulo = nome_variavel.replace("_raw", "").replace("_", " ")
    plt.title(f"Trajetória Clarke (Power Inv.)\n{titulo}")
    plt.xlabel(r"$i_\alpha$")
    plt.ylabel(r"$i_\beta$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(caminho_salvar, format="svg", facecolor="white")
    plt.close()


def padronizar_dados(f_handle, var_name):
    """
    Lê do HDF5 e garante formato (N, 3).
    CORREÇÃO: Preenche fases inexistentes com 0.0 em vez de repetir a anterior.
    """
    if var_name not in f_handle:
        return None
    d = np.array(f_handle[var_name])

    # Ajuste dimensional (garante N linhas)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    elif d.ndim == 2 and d.shape[0] < d.shape[1]:
        d = d.T

    linhas, col = d.shape
    d_out = np.zeros((linhas, 3))  # Inicia tudo com zero (terra/inexistente)

    # Mapeamento correto assumindo ordem sequencial (A -> B -> C)
    if col == 1:
        # Monofásico (A): Fase A tem dados, B e C são 0
        d_out[:, 0] = d[:, 0]
        # d_out[:, 1] e [:, 2] continuam 0.0

    elif col == 2:
        # Bifásico (A, B): A e B têm dados, C é 0
        d_out[:, 0] = d[:, 0]
        d_out[:, 1] = d[:, 1]
        # d_out[:, 2] continua 0.0

    else:
        # Trifásico completo
        d_out[:, 0:3] = d[:, 0:3]

    return d_out

# ----------------------------------------------------------------------
# 5. DADOS E SELEÇÃO DE VARIÁVEIS
# ----------------------------------------------------------------------

TODAS_BARRAS = [
    ("V_800_raw", "Tensão (V)", "V_barra_800"),
    ("V_T2F_raw", "Tensão (V)", "V_barra_T2F"),
    ("V_T2F1_raw", "Tensão (V)", "V_barra_T2F1"),
    ("V_818_raw", "Tensão (V)", "V_barra_818"),
    ("V_820_raw", "Tensão (V)", "V_barra_820"),
    ("V_822_raw", "Tensão (V)", "V_barra_822"),
    ("I_800_raw", "Corrente (A)", "I_barra_800"),
    ("I_T2F_raw", "Corrente (A)", "I_barra_T2F"),
    ("I_T2F1_raw", "Corrente (A)", "I_barra_T2F1"),
    ("I_818_raw", "Corrente (A)", "I_barra_818"),
    ("I_820_raw", "Corrente (A)", "I_barra_820"),
    ("I_822_raw", "Corrente (A)", "I_barra_822"),
]

BARRAS_ATIVAS = [
    "V_800_raw", "V_T2F_raw", "V_T2F1_raw", "V_818_raw", "V_820_raw", "V_822_raw",
    "I_800_raw", "I_T2F_raw", "I_T2F1_raw", "I_818_raw", "I_820_raw", "I_822_raw",
]

todas_vars = [item for item in TODAS_BARRAS if item[0] in BARRAS_ATIVAS]

lista_arquivos_cap5 = [
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao__R_822_-_Falta_AB_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao__R_822_-_Falta_ABC_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao__R_822_-_Falta_AC_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao__R_822_-_Falta_BC_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao__Sem_Falta_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_sem_terra__R_822_-_Falta_AB_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_sem_terra__R_822_-_Falta_ABC_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_sem_terra__R_822_-_Falta_AC_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_sem_terra__R_822_-_Falta_BC_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_sem_terra__Sem_Falta_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR__R_822_-_Falta_AB_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR__R_822_-_Falta_ABC_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR__R_822_-_Falta_AC_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR__R_822_-_Falta_BC_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR__Sem_Falta_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR_sem_terra__R_822_-_Falta_AB_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR_sem_terra__R_822_-_Falta_ABC_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR_sem_terra__R_822_-_Falta_AC_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR_sem_terra__R_822_-_Falta_BC_py.mat',
r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR_sem_terra__Sem_Falta_py.mat',
]

# PARÂMETROS CONFIGURADOS
RELIGADORES_PARAMS = {
    "R_Montante_T2F": {"Pickup": 20, "TMS_TD": 0.3},
    "R_Jusante_T2F1": {"Pickup": 35, "TMS_TD": 0.3}
}

# ----------------------------------------------------------------------
# 6. LOOP PRINCIPAL
# ----------------------------------------------------------------------
def main():
    pasta_tcc = PASTA_RAIZ_RESULTADOS / "Curvas_TCC"
    pasta_tcc.mkdir(parents=True, exist_ok=True)
    plotar_curvas_tcc_religadores(pasta_tcc / "Fig_TCC_IEC_IEEE.svg")
    print(f"[INFO] Curvas TCC geradas em {pasta_tcc}")
    print(f"\nIniciando processamento de {len(lista_arquivos_cap5)} arquivos...\n")

    for caminho_str in lista_arquivos_cap5:
        path_arquivo = Path(caminho_str)
        if not path_arquivo.exists():
            print(f"[AVISO] Arquivo não encontrado: {path_arquivo.name}")
            continue

        try:
            pasta_final = PASTA_RAIZ_RESULTADOS / path_arquivo.stem
            pasta_final.mkdir(parents=True, exist_ok=True)
            print(f"--> Processando: {path_arquivo.name}")

            with h5py.File(path_arquivo, "r") as f:
                t = np.array(f["t"]).flatten()

                # --- 1. GERAÇÃO DE GRÁFICOS (VARIAVEIS DE INTERESSE) ---
                for var_mat, label_y, sufixo in todas_vars:
                    dados = padronizar_dados(f, var_mat)
                    if dados is None or np.max(np.abs(dados)) <= 1e-9: continue

                    plotar_envelope_rms_maximo(t, dados, pasta_final / f"rms_env_{sufixo}.svg", label_y.split(" ")[0])
                    plotar_rms_tres_fases_com_zoom(t, dados, var_mat, pasta_final / f"rms_zoom_{sufixo}.svg",
                                                   label_y.split(" ")[0])
                    gerar_grafico_fft_espectro(t, dados, pasta_final / f"fft_{sufixo}.svg", label_y)
                    seq0, seq1, seq2 = calcular_componentes_simetricas_tempo(dados, t)
                    plotar_sequencias_simetricas(t, seq0, seq1, seq2, var_mat, pasta_final / f"seq_{sufixo}.svg",
                                                 label_y.split(" ")[0])

                    if "Tensão" in label_y:
                        v1, v3 = calcular_v1_v3_global(dados[:, 0], t)
                        if isfinite(v1) and isfinite(v3):
                            plotar_barras_harmonicas_v1v3(v1, v3, var_mat, pasta_final / f"harm_{sufixo}.svg")

                    plotar_trajetoria_clarke(t, dados, var_mat, pasta_final / f"clarke_{sufixo}.svg")

                # --- 2. ANÁLISE DE PROTEÇÃO (TESTE DE TODAS AS CURVAS) ---

                # Leitura das correntes
                dados_m = padronizar_dados(f, "I_T2F_raw")  # Montante
                dados_j = padronizar_dados(f, "I_T2F1_raw")  # Jusante

                icc_m = np.max(get_imax_envelope_vetor(t, dados_m)) if dados_m is not None else 0.0
                icc_j = np.max(get_imax_envelope_vetor(t, dados_j)) if dados_j is not None else 0.0

                # Arquivo CSV de proteção
                csv_path = pasta_final / "analise_protecao_completa.csv"
                with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file, delimiter=';')

                    # Cabeçalho Detalhado
                    header = [
                        "Curva_Teste",
                        "R_Montante(A)", "R_Jusante(A)",
                        "Pickup_M", "Pickup_J",
                        "Tempo_Montante(s)", "Tempo_Jusante(s)",
                        "Diferenca(s)",
                        "1o_A_Atuar", "2o_A_Atuar",
                        "Status_Coordenacao"
                    ]
                    writer.writerow(header)

                    # Configuração Base
                    pm = RELIGADORES_PARAMS["R_Montante_T2F"]
                    pj = RELIGADORES_PARAMS["R_Jusante_T2F1"]

                    # Loop por todos os tipos de curva disponíveis
                    for tipo_curva in CURVAS_DEFS.keys():
                        # Calcula tempos
                        # M = Icc / Pickup
                        mult_m = icc_m / pm["Pickup"]
                        mult_j = icc_j / pj["Pickup"]

                        tm = calcular_tempo_curva(mult_m, pm["TMS_TD"], tipo_curva)
                        tj = calcular_tempo_curva(mult_j, pj["TMS_TD"], tipo_curva)

                        # Lógica de quem atuou
                        atuou_m = isfinite(tm)
                        atuou_j = isfinite(tj)

                        primeiro = "-"
                        segundo = "-"
                        diff = np.inf
                        status = "Nenhum Atuou"

                        if atuou_m and atuou_j:
                            diff = abs(tm - tj)
                            if tm < tj:
                                primeiro = "Montante"
                                segundo = "Jusante"
                                status = "FALHA: Montante Rápido Demais"
                            elif tj < tm:
                                primeiro = "Jusante"
                                segundo = "Montante"
                                status = "OK: Jusante Primeiro"
                            else:
                                primeiro = "Simultaneo"
                                segundo = "Simultaneo"
                                status = "Simultaneo"
                        elif atuou_m and not atuou_j:
                            primeiro = "Montante"
                            segundo = "-"
                            status = "Apenas Montante (Jusante Insensivel)"
                        elif not atuou_m and atuou_j:
                            primeiro = "Jusante"
                            segundo = "-"
                            status = "Apenas Jusante (Montante Insensivel)"

                        # Formatação para CSV
                        s_tm = f"{tm:.4f}" if atuou_m else "Inf"
                        s_tj = f"{tj:.4f}" if atuou_j else "Inf"
                        s_diff = f"{diff:.4f}" if isfinite(diff) else "-"

                        writer.writerow([
                            tipo_curva,
                            f"{icc_m:.2f}", f"{icc_j:.2f}",
                            pm["Pickup"], pj["Pickup"],
                            s_tm, s_tj,
                            s_diff,
                            primeiro, segundo,
                            status
                        ])

        except Exception as e:
            print(f"[ERRO] Falha ao processar {path_arquivo.name}: {e}")

    print("\n--- Processamento Capítulo 5 Finalizado ---")


if __name__ == "__main__":
    main()