#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
from pathlib import Path
from scipy.ndimage import uniform_filter1d
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Configurações de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Cores padrão brasileiro
CORES = {
    'A': '#00FFFF',  # Ciano
    'B': '#FF3232',  # Vermelho
    'C': '#00FF00',  # Verde
    'Ref': '#969696'
}


# Configurações de alta qualidade para publicação acadêmica
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 600  # Alta resolução para impressão
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['legend.framealpha'] = 0.95
mpl.rcParams['legend.edgecolor'] = 'black'
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['axes.grid'] = True

CORES_TESE = {
    'A': '#0066CC',
    'B': '#CC0000',
    'C': '#009900',
    'T2F': '#FF6600',
    'MRT': '#6600CC',
    'Com_Terra': '#008B8B',
    'Sem_Terra': '#B8860B',
}

print("✅ Configurações de alta qualidade para tese aplicadas!")
print("📊 DPI de salvamento: 600 (qualidade publicação)")

#%%


# Barras disponíveis
BARRAS_DISPONIVEIS = ["800", "818", "820", "822", "T2F1", "T2F"]

# Configuração
FREQ = 60  # Hz

print("✅ Bibliotecas carregadas com sucesso!")
print(f"📊 Barras disponíveis: {BARRAS_DISPONIVEIS}")


#%%
# 🔧 CONFIGURE AQUI SEUS CAMINHOS
PASTA_DADOS = (
    "C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/"
    "T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/"
    "Teste_Novo_Sem_Terra_13/Processados_HDF5/"
)  # Onde estão seus arquivos .mat

PASTA_SAIDA = "./resultados_analise_13"  # Onde salvar figuras

# Cria pasta de saída
Path(PASTA_SAIDA).mkdir(parents=True, exist_ok=True)

print("✅ Configuração de pastas:")
print(f"   📂 Dados: {PASTA_DADOS}")
print(f"   💾 Saída: {PASTA_SAIDA}")


#%%
def carregar_mat(caminho):
    """Carrega arquivo .mat (v7.3 HDF5 ou formato antigo)."""
    try:
        with h5py.File(caminho, 'r') as f:
            dados = {}
            for key in f.keys():
                if not key.startswith('__'):
                    try:
                        item = f[key]
                        if isinstance(item, h5py.Dataset):
                            dados[key] = item[()]
                            if isinstance(dados[key], np.ndarray) and dados[key].ndim == 2:
                                if dados[key].shape[0] == 3 and dados[key].shape[1] > 3:
                                    dados[key] = dados[key].T
                    except Exception as e:
                        print(f"  ⚠️ Aviso ao ler '{key}': {e}")
            return dados
    except Exception:
        try:
            return sio.loadmat(caminho, squeeze_me=False)
        except Exception as e:
            print(f"❌ Erro ao carregar {caminho}: {e}")
            return None


def identificar_barras_no_arquivo(dados):
    """Identifica barras presentes no arquivo."""
    barras_encontradas = []
    for key in dados.keys():
        if '_raw' in key and not key.startswith('__'):
            partes = key.split('_')
            if len(partes) >= 2:
                barra = partes[1]
                if barra not in barras_encontradas:
                    barras_encontradas.append(barra)
    return sorted(barras_encontradas)


def _garantir_3_fases(arr):
    """Garante array com shape (N, 3)."""
    arr = np.array(arr, dtype=np.float64)
    if arr.ndim == 1:
        return np.column_stack([arr, arr, arr])
    if arr.shape[0] == 3 and arr.shape[1] > 3:
        return arr.T
    if arr.shape[1] == 3:
        return arr
    if arr.ndim >= 2 and arr.shape[1] >= 3:
        return arr[:, :3]
    if arr.ndim >= 2:
        col = arr[:, 0]
        return np.column_stack([col, col, col])
    return arr


def extrair_dados_barra(dados, barra):
    """Extrai t, V, I de uma barra específica."""
    chaves_v = [f'V_{barra}_raw', f'V_{barra}', f'v_{barra}_raw']
    chaves_i = [f'I_{barra}_raw', f'I_{barra}', f'i_{barra}_raw']
    chaves_t = ['t', 'time', 'tempo']

    t = None
    for key in chaves_t:
        if key in dados:
            t = np.array(dados[key]).flatten()
            break

    V = None
    for key in chaves_v:
        if key in dados:
            V = np.array(dados[key])
            break

    I = None
    for key in chaves_i:
        if key in dados:
            I = np.array(dados[key])
            break

    if t is None or V is None or I is None:
        return None, None, None

    V = _garantir_3_fases(V)
    I = _garantir_3_fases(I)

    L = min(len(t), len(V), len(I))
    t = t[:L]
    V = V[:L, :]
    I = I[:L, :]

    return t, V, I


print("✅ Funções de carregamento definidas!")


#%%
class ProcessadorSinais:
    """Processa sinais de tensão e corrente de uma barra."""

    def __init__(self, t, V, I, freq=60, barra_nome=""):
        self.t = np.array(t).flatten()
        self.v = np.array(V)
        self.i = np.array(I)
        self.freq = freq
        self.barra_nome = barra_nome

        self.dt = float(self.t[1] - self.t[0]) if len(self.t) > 1 else 1/60
        self.fs = 1.0 / self.dt
        self.samples_per_cycle = int(self.fs / self.freq)

        print(f"  ✅ Barra {barra_nome}: {len(self.t)} amostras, fs={self.fs:.1f} Hz")

    def calcular_rms(self, sinal, janela=None):
        """RMS móvel."""
        if janela is None:
            janela = self.samples_per_cycle
        janela = max(1, int(janela))
        return np.sqrt(uniform_filter1d(sinal**2, janela, axis=0))

    def clarke(self):
        """Transformada de Clarke."""
        a, b, c = self.i[:, 0], self.i[:, 1], self.i[:, 2]
        alpha = (2*a - b - c) / 3.0
        beta = (b - c) / np.sqrt(3.0)
        return alpha, beta

    def componentes_simetricas(self, idx):
        """Componentes simétricas em um índice de tempo."""
        ia, ib, ic = self.i[idx, 0], self.i[idx, 1], self.i[idx, 2]
        a_op = np.exp(1j * 2*np.pi/3)

        I0 = np.abs((ia + ib + ic) / 3.0)
        I1 = np.abs((ia + a_op*ib + a_op**2*ic) / 3.0)
        I2 = np.abs((ia + a_op**2*ib + a_op*ic) / 3.0)

        return I0, I1, I2

    def fft_espectro(self, fase_idx, n_cycles=1.0):
        """FFT com janela de Hann."""
        n = int(max(16, round((self.fs / self.freq) * n_cycles)))
        idx_fim = len(self.i) - 1
        idx_inicio = max(0, idx_fim - n + 1)

        seg = self.i[idx_inicio:idx_fim+1, fase_idx]
        seg = seg - np.mean(seg)

        w = np.hanning(len(seg))
        segw = seg * w

        X = np.fft.rfft(segw)
        freqs = np.fft.rfftfreq(len(segw), d=self.dt)

        w_sum = np.sum(w) if np.sum(w) != 0 else 1.0
        mag = (2.0 / w_sum) * np.abs(X)
        if len(mag) > 0:
            mag[0] = mag[0] / 2.0

        return freqs, mag

    def estatisticas(self):
        """Estatísticas básicas das correntes e tensões RMS."""
        i_rms = self.calcular_rms(self.i)
        v_rms = self.calcular_rms(self.v)
        return {
            'barra': self.barra_nome,
            'i_max_A': np.max(np.abs(i_rms[:, 0])),
            'i_max_B': np.max(np.abs(i_rms[:, 1])),
            'i_max_C': np.max(np.abs(i_rms[:, 2])),
            'i_media_A': np.mean(np.abs(i_rms[:, 0])),
            'i_media_B': np.mean(np.abs(i_rms[:, 1])),
            'i_media_C': np.mean(np.abs(i_rms[:, 2])),
            'v_max_A': np.max(np.abs(v_rms[:, 0])),
            'v_max_B': np.max(np.abs(v_rms[:, 1])),
            'v_max_C': np.max(np.abs(v_rms[:, 2])),
        }


print("✅ Classe ProcessadorSinais definida!")

def plotar_tensao_evento_falta(proc, titulo, instante_falta=0.5/3, salvar_como=None):
    """
    Plota tensão destacando o momento da falta.
    Ideal para mostrar afundamento de tensão.
    """
    v_rms = proc.calcular_rms(proc.v)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(proc.t, v_rms[:, 0], color=CORES_TESE['A'], linewidth=2.5,
            label='Fase A', alpha=0.9)
    ax.plot(proc.t, v_rms[:, 1], color=CORES_TESE['B'], linewidth=2.5,
            label='Fase B', alpha=0.9)
    ax.plot(proc.t, v_rms[:, 2], color=CORES_TESE['C'], linewidth=2.5,
            label='Fase C', alpha=0.9)

    ax.axvline(x=instante_falta, color='red', linestyle='--', linewidth=2,
               label='Instante da Falta', alpha=0.7)

    ax.axvspan(0, instante_falta, alpha=0.5/3, color='green', label='Operação Normal')
    ax.axvspan(instante_falta, proc.t[-1], alpha=0.5/3, color='red', label='Durante Falta')

    ax.set_xlabel('Tempo (s)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Tensão RMS (V)', fontweight='bold', fontsize=14)
    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', ncol=2, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    v_antes = np.mean(v_rms[proc.t < instante_falta, 0])
    v_depois = np.min(v_rms[proc.t > instante_falta, 0])
    queda = ((v_antes - v_depois) / v_antes) * 100

    ax.text(0.98, 0.97, f'Queda de Tensão: {queda:.1f}%',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if salvar_como:
        plt.savefig(salvar_como, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"  💾 Salvo em alta resolução: {salvar_como}")

    #plt.show()

print("✅ Função plotar_tensao_evento_falta definida!")

#%%
def plotar_rms_multiplas_barras(processadores, titulo, salvar_como=None):
    """Plota RMS de tensão e corrente para múltiplas barras."""
    n_barras = len(processadores)
    fig, axes = plt.subplots(n_barras, 2, figsize=(16, 5*n_barras), sharex=True)

    if n_barras == 1:
        axes = axes.reshape(1, -1)

    for idx, (barra, proc) in enumerate(processadores.items()):
        v_rms = proc.calcular_rms(proc.v)
        i_rms = proc.calcular_rms(proc.i)

        ax_v = axes[idx, 0]
        ax_v.plot(proc.t, v_rms[:, 0], color=CORES['A'], linewidth=1.5, label='Fase A', alpha=0.9)
        ax_v.plot(proc.t, v_rms[:, 1], color=CORES['B'], linewidth=1.5, label='Fase B', alpha=0.9)
        ax_v.plot(proc.t, v_rms[:, 2], color=CORES['C'], linewidth=1.5, label='Fase C', alpha=0.9)
        ax_v.grid(True, alpha=0.3)
        ax_v.set_ylabel(f'Tensão RMS (V)\nBarra {barra}', fontweight='bold')
        ax_v.legend(loc='best', fontsize=9)
        if idx == 0:
            ax_v.set_title('Tensão RMS', fontweight='bold', fontsize=14)

        ax_i = axes[idx, 1]
        ax_i.plot(proc.t, i_rms[:, 0], color=CORES['A'], linewidth=1.5, label='Fase A', alpha=0.9)
        ax_i.plot(proc.t, i_rms[:, 1], color=CORES['B'], linewidth=1.5, label='Fase B', alpha=0.9)
        ax_i.plot(proc.t, i_rms[:, 2], color=CORES['C'], linewidth=1.5, label='Fase C', alpha=0.9)
        ax_i.grid(True, alpha=0.3)
        ax_i.set_ylabel(f'Corrente RMS (A)\nBarra {barra}', fontweight='bold')
        ax_i.legend(loc='best', fontsize=9)
        if idx == 0:
            ax_i.set_title('Corrente RMS', fontweight='bold', fontsize=14)

    axes[-1, 0].set_xlabel('Tempo (s)', fontweight='bold')
    axes[-1, 1].set_xlabel('Tempo (s)', fontweight='bold')

    fig.suptitle(titulo, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if salvar_como:
        plt.savefig(salvar_como, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  💾 Salvo: {salvar_como}")

    #plt.show()

print("✅ Função plotar_rms_multiplas_barras definida!")


#%%
def plotar_clarke_multiplas_barras(processadores, titulo, salvar_como=None):
    """Plota Clarke para múltiplas barras."""
    n_barras = len(processadores)
    cols = min(3, n_barras)
    rows = (n_barras + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    axes = axes.flatten() if n_barras > 1 else [axes]

    for idx, (barra, proc) in enumerate(processadores.items()):
        alpha, beta = proc.clarke()
        ax = axes[idx]

        ax.plot(alpha, beta, 'b-', linewidth=0.5, alpha=0.3)

        n_destaque = min(500, len(alpha))
        ax.plot(alpha[-n_destaque:], beta[-n_destaque:],
                color='cyan', linewidth=2, alpha=0.8)

        ax.plot(alpha[-1], beta[-1], 'ro', markersize=12,
                markerfacecolor='red', markeredgecolor='darkred',
                markeredgewidth=2, zorder=10)

        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlabel('α (A)', fontweight='bold')
        ax.set_ylabel('β (A)', fontweight='bold')
        ax.set_title(f'Barra {barra}', fontweight='bold')

    for idx in range(len(processadores), len(axes)):
        axes[idx].remove()

    fig.suptitle(titulo, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if salvar_como:
        plt.savefig(salvar_como, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  💾 Salvo: {salvar_como}")

    #plt.show()

print("✅ Função plotar_clarke_multiplas_barras definida!")


#%%
def plotar_comparacao_correntes_barras(processadores, titulo, salvar_como=None):
    """Compara correntes máximas entre barras."""
    barras = list(processadores.keys())
    dados_comp = [proc.estatisticas() for proc in processadores.values()]

    i_max_A = [d['i_max_A'] for d in dados_comp]
    i_max_B = [d['i_max_B'] for d in dados_comp]
    i_max_C = [d['i_max_C'] for d in dados_comp]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(barras))
    width = 0.25

    ax.bar(x - width, i_max_A, width, label='Fase A', color=CORES['A'],
           edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.bar(x, i_max_B, width, label='Fase B', color=CORES['B'],
           edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.bar(x + width, i_max_C, width, label='Fase C', color=CORES['C'],
           edgecolor='black', linewidth=1.5, alpha=0.85)

    for i, (a, b, c) in enumerate(zip(i_max_A, i_max_B, i_max_C)):
        ax.text(i - width, a + a*0.02, f'{a:.0f}', ha='center', va='bottom', fontsize=9)
        ax.text(i, b + b*0.02, f'{b:.0f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width, c + c*0.02, f'{c:.0f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Barra', fontweight='bold', fontsize=13)
    ax.set_ylabel('Corrente Máxima (A)', fontweight='bold', fontsize=13)
    ax.set_title(titulo, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(barras)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if salvar_como:
        plt.savefig(salvar_como, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  💾 Salvo: {salvar_como}")

    #plt.show()


def plotar_componentes_simetricas_multiplas(processadores, titulo, salvar_como=None):
    """Plota componentes simétricas para múltiplas barras."""
    fig, ax = plt.subplots(figsize=(14, 7))

    barras = list(processadores.keys())
    n_barras = len(barras)

    I0_vals = []
    I1_vals = []
    I2_vals = []

    for proc in processadores.values():
        idx_meio = len(proc.t) // 2
        I0, I1, I2 = proc.componentes_simetricas(idx_meio)
        I0_vals.append(I0)
        I1_vals.append(I1)
        I2_vals.append(I2)

    x = np.arange(n_barras)
    width = 0.25

    ax.bar(x - width, I0_vals, width, label='Sequência Zero (I₀)',
           color='#E74C3C', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.bar(x, I1_vals, width, label='Sequência Positiva (I₁)',
           color='#3498DB', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.bar(x + width, I2_vals, width, label='Sequência Negativa (I₂)',
           color='#F39C12', edgecolor='black', linewidth=1.5, alpha=0.85)

    ax.set_xlabel('Barra', fontweight='bold', fontsize=13)
    ax.set_ylabel('Magnitude (A)', fontweight='bold', fontsize=13)
    ax.set_title(titulo, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(barras)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if salvar_como:
        plt.savefig(salvar_como, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  💾 Salvo: {salvar_como}")

    #plt.show()

print("✅ Funções de comparação definidas!")


#%%
def criar_tabela_estatisticas(processadores, salvar_como=None):
    """Cria tabela de estatísticas por barra."""
    dados_lista = []

    for barra, proc in processadores.items():
        stats = proc.estatisticas()
        dados_lista.append(stats)

    df = pd.DataFrame(dados_lista)

    print("\n" + "="*80)
    print("📊 TABELA DE ESTATÍSTICAS")
    print("="*80)

    colunas_numericas = [col for col in df.columns if col != 'barra']
    for col in colunas_numericas:
        df[col] = df[col].round(2)

    print(df.to_string(index=False))
    print("="*80)

    if salvar_como:
        df.to_csv(salvar_como, index=False)
        print(f"💾 Tabela salva: {salvar_como}")

    return df

print("✅ Função criar_tabela_estatisticas definida!")

def plotar_corrente_falta_zoom(proc, titulo, instante_falta=0.5/3, janela_zoom=0.05, salvar_como=None):
    """
    Plota corrente de falta com zoom na região crítica.
    """
    i_rms = proc.calcular_rms(proc.i)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax_main = fig.add_subplot(gs[0:2, :])
    ax_main.plot(proc.t, i_rms[:, 0], color=CORES_TESE['A'], linewidth=2.5, label='Fase A')
    ax_main.plot(proc.t, i_rms[:, 1], color=CORES_TESE['B'], linewidth=2.5, label='Fase B')
    ax_main.plot(proc.t, i_rms[:, 2], color=CORES_TESE['C'], linewidth=2.5, label='Fase C')
    ax_main.axvline(x=instante_falta, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_main.set_xlabel('Tempo (s)', fontweight='bold', fontsize=14)
    ax_main.set_ylabel('Corrente RMS (A)', fontweight='bold', fontsize=14)
    ax_main.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
    ax_main.legend(loc='best', fontsize=12)
    ax_main.grid(True, alpha=0.3)

    t_zoom_start = instante_falta - janela_zoom/2
    t_zoom_end = instante_falta + janela_zoom/2
    from matplotlib.patches import Rectangle
    rect = Rectangle((t_zoom_start, ax_main.get_ylim()[0]),
                      janela_zoom, ax_main.get_ylim()[1] - ax_main.get_ylim()[0],
                      linewidth=2, edgecolor='orange', facecolor='orange', alpha=0.2)
    ax_main.add_patch(rect)

    ax_zoom1 = fig.add_subplot(gs[2, 0])
    mask_antes = (proc.t >= t_zoom_start - 0.02) & (proc.t <= instante_falta)
    ax_zoom1.plot(proc.t[mask_antes], i_rms[mask_antes, 0], color=CORES_TESE['A'], linewidth=2)
    ax_zoom1.plot(proc.t[mask_antes], i_rms[mask_antes, 1], color=CORES_TESE['B'], linewidth=2)
    ax_zoom1.plot(proc.t[mask_antes], i_rms[mask_antes, 2], color=CORES_TESE['C'], linewidth=2)
    ax_zoom1.axvline(x=instante_falta, color='red', linestyle='--', linewidth=1.5)
    ax_zoom1.set_title('Zoom: Antes da Falta', fontsize=12, fontweight='bold')
    ax_zoom1.set_xlabel('Tempo (s)', fontsize=11)
    ax_zoom1.set_ylabel('Corrente (A)', fontsize=11)
    ax_zoom1.grid(True, alpha=0.3)

    ax_zoom2 = fig.add_subplot(gs[2, 1])
    mask_durante = (proc.t >= instante_falta) & (proc.t <= t_zoom_end + 0.02)
    ax_zoom2.plot(proc.t[mask_durante], i_rms[mask_durante, 0], color=CORES_TESE['A'], linewidth=2)
    ax_zoom2.plot(proc.t[mask_durante], i_rms[mask_durante, 1], color=CORES_TESE['B'], linewidth=2)
    ax_zoom2.plot(proc.t[mask_durante], i_rms[mask_durante, 2], color=CORES_TESE['C'], linewidth=2)
    ax_zoom2.axvline(x=instante_falta, color='red', linestyle='--', linewidth=1.5)
    ax_zoom2.set_title('Zoom: Durante a Falta', fontsize=12, fontweight='bold')
    ax_zoom2.set_xlabel('Tempo (s)', fontsize=11)
    ax_zoom2.set_ylabel('Corrente (A)', fontsize=11)
    ax_zoom2.grid(True, alpha=0.3)

    i_max = np.max(i_rms)
    ax_main.text(0.02, 0.97, f'Corrente Máx: {i_max:.1f} A',
                 transform=ax_main.transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    if salvar_como:
        plt.savefig(salvar_como, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"  💾 Salvo: {salvar_como}")

    #plt.show()

print("✅ Função plotar_corrente_falta_zoom definida!")


#%%
def plotar_clarke_profissional(proc, titulo, salvar_como=None):
    """
    Clarke profissional com análise de circularidade.
    """
    alpha, beta = proc.clarke()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(alpha, beta, color='#1f77b4', linewidth=0.8, alpha=0.4, label='Trajetória')

    n_destaque = min(300, len(alpha))
    ax1.plot(alpha[-n_destaque:], beta[-n_destaque:],
             color='#ff7f0e', linewidth=2.5, alpha=0.9, label='Últimos ciclos')

    ax1.plot(alpha[-1], beta[-1], 'ro', markersize=15,
             markerfacecolor='red', markeredgecolor='darkred',
             markeredgewidth=2.5, zorder=10, label='Ponto atual')

    i_rms = proc.calcular_rms(proc.i)
    raio_ref = np.mean(np.mean(i_rms, axis=0))
    circle = plt.Circle((0, 0), raio_ref, color='green', fill=False,
                        linestyle='--', linewidth=2, alpha=0.5, label='Referência Balanceada')
    ax1.add_patch(circle)

    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axis('equal')
    ax1.set_xlabel('Componente α (A)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Componente β (A)', fontweight='bold', fontsize=14)
    ax1.set_title(f'Plano α-β - {titulo}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)

    magnitude = np.sqrt(alpha**2 + beta**2)
    ax2.plot(proc.t, magnitude, color='#2ca02c', linewidth=2, label='|I| = √(α² + β²)')
    ax2.axhline(y=raio_ref, color='green', linestyle='--', linewidth=2,
                alpha=0.5, label='Referência')
    ax2.fill_between(proc.t, 0, magnitude, alpha=0.2, color='#2ca02c')

    ax2.set_xlabel('Tempo (s)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Magnitude da Corrente (A)', fontweight='bold', fontsize=14)
    ax2.set_title('Evolução Temporal da Magnitude', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if salvar_como:
        plt.savefig(salvar_como, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"  💾 Salvo: {salvar_como}")

    #plt.show()

print("✅ Função plotar_clarke_profissional definida!")


#%%
def comparar_t2f_mrt_mesma_barra(proc_t2f, proc_mrt, barra, tipo_falta, salvar_como=None):
    """
    Compara T2F vs MRT para a mesma barra e tipo de falta.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    i_rms_t2f = proc_t2f.calcular_rms(proc_t2f.i)
    i_rms_mrt = proc_mrt.calcular_rms(proc_mrt.i)

    ax1.plot(proc_t2f.t, i_rms_t2f[:, 0], color=CORES_TESE['T2F'],
             linewidth=2.5, label='T2F - Fase A', linestyle='-')
    ax1.plot(proc_mrt.t, i_rms_mrt[:, 0], color=CORES_TESE['MRT'],
             linewidth=2.5, label='MRT - Fase A', linestyle='--')
    ax1.set_ylabel('Corrente RMS Fase A (A)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Tempo (s)', fontweight='bold', fontsize=12)
    ax1.set_title(f'Comparação Corrente - Barra {barra}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    v_rms_t2f = proc_t2f.calcular_rms(proc_t2f.v)
    v_rms_mrt = proc_mrt.calcular_rms(proc_mrt.v)

    ax2.plot(proc_t2f.t, v_rms_t2f[:, 0], color=CORES_TESE['T2F'],
             linewidth=2.5, label='T2F - Fase A', linestyle='-')
    ax2.plot(proc_mrt.t, v_rms_mrt[:, 0], color=CORES_TESE['MRT'],
             linewidth=2.5, label='MRT - Fase A', linestyle='--')
    ax2.set_ylabel('Tensão RMS Fase A (V)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Tempo (s)', fontweight='bold', fontsize=12)
    ax2.set_title(f'Comparação Tensão - Barra {barra}', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    categorias = ['Corrente\nMáxima (A)', 'Tensão\nMínima (V)', 'Corrente\nMédia (A)']

    i_max_t2f = np.max(i_rms_t2f[:, 0])
    i_max_mrt = np.max(i_rms_mrt[:, 0])
    v_min_t2f = np.min(v_rms_t2f[:, 0])
    v_min_mrt = np.min(v_rms_mrt[:, 0])
    i_med_t2f = np.mean(i_rms_t2f[:, 0])
    i_med_mrt = np.mean(i_rms_mrt[:, 0])

    valores_t2f = [i_max_t2f, v_min_t2f, i_med_t2f]
    valores_mrt = [i_max_mrt, v_min_mrt, i_med_mrt]

    x = np.arange(len(categorias))
    width = 0.35

    bars1 = ax3.bar(x - width/2, valores_t2f, width, label='T2F',
                    color=CORES_TESE['T2F'], edgecolor='black', linewidth=1.5, alpha=0.85)
    bars2 = ax3.bar(x + width/2, valores_mrt, width, label='MRT',
                    color=CORES_TESE['MRT'], edgecolor='black', linewidth=1.5, alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.0f}', ha='center', va='bottom', fontsize=10)

    ax3.set_ylabel('Valor', fontweight='bold', fontsize=12)
    ax3.set_title('Comparação de Indicadores', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categorias, fontsize=10)
    ax3.legend(fontsize=11)
    ax3.grid(True, axis='y', alpha=0.3)

    ax4 = axes[1, 1]
    diferencas = [((valores_t2f[i] - valores_mrt[i]) / valores_mrt[i] * 100)
                  for i in range(len(valores_t2f))]

    colors_diff = ['red' if d > 0 else 'green' for d in diferencas]
    bars_diff = ax4.bar(categorias, diferencas, color=colors_diff,
                        edgecolor='black', linewidth=1.5, alpha=0.7)

    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Diferença Percentual (%)', fontweight='bold', fontsize=12)
    ax4.set_title('T2F vs MRT - Diferença Relativa', fontsize=13, fontweight='bold')
    ax4.set_xticklabels(categorias, fontsize=10)
    ax4.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars_diff, diferencas):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:+.1f}%', ha='center',
                 va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')

    fig.suptitle(f'T2F vs MRT - Falta {tipo_falta} - Barra {barra}',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    if salvar_como:
        plt.savefig(salvar_como, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"  💾 Salvo: {salvar_como}")

    #plt.show()

print("✅ Função comparar_t2f_mrt_mesma_barra definida!")


#%%
def comparar_com_sem_terra(proc_com_terra, proc_sem_terra, barra, tipo_falta, salvar_como=None):
    """
    Compara configurações com e sem aterramento.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    i_rms_com = proc_com_terra.calcular_rms(proc_com_terra.i)
    i_rms_sem = proc_sem_terra.calcular_rms(proc_sem_terra.i)
    v_rms_com = proc_com_terra.calcular_rms(proc_com_terra.v)
    v_rms_sem = proc_sem_terra.calcular_rms(proc_sem_terra.v)

    for fase_idx, fase_nome in enumerate(['A', 'B', 'C']):
        row = fase_idx

        ax_i = axes[row, 0]
        ax_i.plot(proc_com_terra.t, i_rms_com[:, fase_idx],
                  color=CORES_TESE['Com_Terra'], linewidth=2.5,
                  label='Com Aterramento', linestyle='-')
        ax_i.plot(proc_sem_terra.t, i_rms_sem[:, fase_idx],
                  color=CORES_TESE['Sem_Terra'], linewidth=2.5,
                  label='Sem Aterramento', linestyle='--')
        ax_i.set_ylabel(f'Corrente Fase {fase_nome} (A)', fontweight='bold')
        ax_i.set_title(f'Corrente RMS - Fase {fase_nome}', fontweight='bold')
        ax_i.legend(fontsize=10)
        ax_i.grid(True, alpha=0.3)
        if row == 2:
            ax_i.set_xlabel('Tempo (s)', fontweight='bold')

        ax_v = axes[row, 1]
        ax_v.plot(proc_com_terra.t, v_rms_com[:, fase_idx],
                  color=CORES_TESE['Com_Terra'], linewidth=2.5,
                  label='Com Aterramento', linestyle='-')
        ax_v.plot(proc_sem_terra.t, v_rms_sem[:, fase_idx],
                  color=CORES_TESE['Sem_Terra'], linewidth=2.5,
                  label='Sem Aterramento', linestyle='--')
        ax_v.set_ylabel(f'Tensão Fase {fase_nome} (V)', fontweight='bold')
        ax_v.set_title(f'Tensão RMS - Fase {fase_nome}', fontweight='bold')
        ax_v.legend(fontsize=10)
        ax_v.grid(True, alpha=0.3)
        if row == 2:
            ax_v.set_xlabel('Tempo (s)', fontweight='bold')

    fig.suptitle(f'Com vs Sem Aterramento - Falta {tipo_falta} - Barra {barra}',
                 fontsize=18, fontweight='bold', y=0.998)
    plt.tight_layout()

    if salvar_como:
        plt.savefig(salvar_como, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"  💾 Salvo: {salvar_como}")

    #plt.show()

print("✅ Função comparar_com_sem_terra definida!")


#%%
def analisar_arquivo_completo(caminho_arquivo, barras_interesse=None, gerar_graficos=True):
    """Analisa um arquivo .mat completo (todas as barras de interesse)."""
    print("\n" + "="*80)
    print(f"🔬 ANALISANDO: {Path(caminho_arquivo).name}")
    print("="*80)

    dados = carregar_mat(caminho_arquivo)
    if dados is None:
        return None

    barras_disponiveis = identificar_barras_no_arquivo(dados)
    print(f"\n📍 Barras encontradas: {barras_disponiveis}")

    if barras_interesse:
        barras_processar = [b for b in barras_interesse if b in barras_disponiveis]
    else:
        barras_processar = barras_disponiveis

    print(f"⚙️ Processando: {barras_processar}")

    processadores = {}

    for barra in barras_processar:
        print(f"\n🔹 Barra {barra}:")
        t, V, I = extrair_dados_barra(dados, barra)

        if t is None:
            print(f"  ❌ Dados não encontrados para barra {barra}")
            continue

        try:
            proc = ProcessadorSinais(t, V, I, freq=FREQ, barra_nome=barra)
            processadores[barra] = proc
        except Exception as e:
            print(f"  ❌ Erro ao processar barra {barra}: {e}")

    if not processadores:
        print("❌ Nenhuma barra foi processada com sucesso")
        return None

    if gerar_graficos:
        nome_base = Path(caminho_arquivo).stem
        pasta_arquivo = Path(PASTA_SAIDA) / nome_base
        pasta_arquivo.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Gerando gráficos em: {pasta_arquivo}")

        titulo_base = nome_base.replace('_py', '').replace('_', ' ')

        plotar_rms_multiplas_barras(
            processadores,
            f'📊 Tensão e Corrente RMS - {titulo_base}',
            pasta_arquivo / f'{nome_base}_RMS_todas_barras.png'
        )

        plotar_clarke_multiplas_barras(
            processadores,
            f'🔄 Transformada de Clarke - {titulo_base}',
            pasta_arquivo / f'{nome_base}_Clarke_todas_barras.png'
        )

        plotar_comparacao_correntes_barras(
            processadores,
            f'⚡ Comparação de Correntes Máximas - {titulo_base}',
            pasta_arquivo / f'{nome_base}_Comparacao_Correntes.png'
        )

        plotar_componentes_simetricas_multiplas(
            processadores,
            f'🔢 Componentes Simétricas - {titulo_base}',
            pasta_arquivo / f'{nome_base}_Componentes_Simetricas.png'
        )

        criar_tabela_estatisticas(
            processadores,
            pasta_arquivo / f'{nome_base}_Estatisticas.csv'
        )

        for barra, proc in processadores.items():
            # Tensão com evento de falta
            plotar_tensao_evento_falta(
                proc,
                f'Tensão Durante Falta - Barra {barra}',
                instante_falta=0.5 / 3,  # ajuste se seu tempo de falta for outro
                salvar_como=pasta_arquivo / f'{nome_base}_Tensao_Falta_Barra_{barra}.png'
            )

            # Corrente com zoom na falta
            plotar_corrente_falta_zoom(
                proc,
                f'Corrente de Falta (Zoom) - Barra {barra}',
                instante_falta=0.5 / 3,
                janela_zoom=0.05,
                salvar_como=pasta_arquivo / f'{nome_base}_Corrente_Zoom_Barra_{barra}.png'
            )

            # Clarke profissional
            plotar_clarke_profissional(
                proc,
                f'Barra {barra}',
                salvar_como=pasta_arquivo / f'{nome_base}_Clarke_Profissional_Barra_{barra}.png'
            )

    print("\n✅ Análise concluída!")
    return processadores

print("✅ Função analisar_arquivo_completo definida!")


#%%
def analisar_multiplos_arquivos(pasta_dados, padrao='*.mat', barras_interesse=None):
    """Analisa múltiplos arquivos .mat em lote."""
    pasta = Path(pasta_dados)
    arquivos = list(pasta.glob(padrao))

    print("\n" + "="*80)
    print(f"🚀 ANÁLISE EM LOTE")
    print("="*80)
    print(f"📂 Pasta: {pasta}")
    print(f"🔍 Padrão: {padrao}")
    print(f"📄 Encontrados: {len(arquivos)} arquivo(s)")

    resultados = {}

    for idx, arquivo in enumerate(arquivos, 1):
        print(f"\n{'='*80}")
        print(f"Arquivo {idx}/{len(arquivos)}")
        print(f"{'='*80}")

        try:
            proc = analisar_arquivo_completo(arquivo, barras_interesse, gerar_graficos=True)
            if proc:
                resultados[arquivo.name] = proc
        except Exception as e:
            print(f"❌ Erro ao processar {arquivo.name}: {e}")

    print("\n" + "="*80)
    print("✅ ANÁLISE EM LOTE CONCLUÍDA")
    print("="*80)
    print(f"✔️ Arquivos processados: {len(resultados)}/{len(arquivos)}")
    print(f"💾 Resultados salvos em: {PASTA_SAIDA}")

    return resultados

print("✅ Função analisar_multiplos_arquivos definida!")

#%%
# 🎯 ANÁLISE DE UM ARQUIVO ESPECÍFICO

arquivo_teste = (
    "C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/"
    "T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/"
    "Teste_Novo_Sem_Terra_11/Processados_HDF5/"
    "Qualificacao__R_820_-_Falta_ABC_py.mat"
)

processadores = analisar_arquivo_completo(
    arquivo_teste,
    barras_interesse=["800", "818", "820", "822", "T2F", "T2F1"],
    gerar_graficos=True
)

if processadores:
    print(f"\n✅ Processadas {len(processadores)} barras: {list(processadores.keys())}")


#%%
# 🔍 EXPLORAÇÃO: Ver estrutura de um arquivo

arquivo_explorar = (
    "C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/"
    "T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/"
    "Teste_Novo_Sem_Terra_11/Processados_HDF5/"
    "Qualificacao__R_820_-_Falta_ABC_py.mat"
)

dados = carregar_mat(arquivo_explorar)

if dados:
    print("\n📋 ESTRUTURA DO ARQUIVO:")
    print("="*80)

    for chave, valor in dados.items():
        if not chave.startswith('__'):
            if isinstance(valor, np.ndarray):
                print(f"  {chave:20s} : shape={valor.shape} dtype={valor.dtype}")
            else:
                print(f"  {chave:20s} : type={type(valor).__name__}")

    print("\n📍 Barras identificadas:")
    barras = identificar_barras_no_arquivo(dados)
    print(f"   {barras}")


#%%
# 💡 ÁREA DE TESTES PERSONALIZADOS - VERSÃO MELHORADA

if 'processadores' in locals() and processadores and '820' in processadores:
    proc_820 = processadores['820']
    stats = proc_820.estatisticas()
    print(f"\n{'='*60}")
    print(f"📊 ESTATÍSTICAS DETALHADAS - BARRA 820")
    print(f"{'='*60}")

    print(f"\n🔹 Corrente Máxima (A):")
    print(f"   Fase A: {stats['i_max_A']:8.2f} A")
    print(f"   Fase B: {stats['i_max_B']:8.2f} A")
    print(f"   Fase C: {stats['i_max_C']:8.2f} A")

    print(f"\n🔹 Corrente Média (A):")
    print(f"   Fase A: {stats['i_media_A']:8.2f} A")
    print(f"   Fase B: {stats['i_media_B']:8.2f} A")
    print(f"   Fase C: {stats['i_media_C']:8.2f} A")

    print(f"\n🔹 Tensão Máxima (V):")
    print(f"   Fase A: {stats['v_max_A']:8.0f} V")
    print(f"   Fase B: {stats['v_max_B']:8.0f} V")
    print(f"   Fase C: {stats['v_max_C']:8.0f} V")
    print(f"{'='*60}\n")

    i_rms = proc_820.calcular_rms(proc_820.i)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(proc_820.t, i_rms[:, 0], label='Fase A', color=CORES['A'], linewidth=2.5)
    ax1.plot(proc_820.t, i_rms[:, 1], label='Fase B', color=CORES['B'], linewidth=2.5)
    ax1.plot(proc_820.t, i_rms[:, 2], label='Fase C', color=CORES['C'], linewidth=2.5)
    ax1.set_ylabel('Corrente RMS (A)', fontweight='bold', fontsize=13)
    ax1.set_title('Corrente RMS - Barra 820', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    i_max_global = np.max(i_rms)
    ax1.axhline(y=i_max_global, color='red', linestyle='--',
                linewidth=1.5, alpha=0.6, label=f'Máximo: {i_max_global:.1f} A')
    ax1.legend(loc='best', fontsize=11)

    v_rms = proc_820.calcular_rms(proc_820.v)
    ax2.plot(proc_820.t, v_rms[:, 0], label='Fase A', color=CORES['A'], linewidth=2.5)
    ax2.plot(proc_820.t, v_rms[:, 1], label='Fase B', color=CORES['B'], linewidth=2.5)
    ax2.plot(proc_820.t, v_rms[:, 2], label='Fase C', color=CORES['C'], linewidth=2.5)
    ax2.set_xlabel('Tempo (s)', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Tensão RMS (V)', fontweight='bold', fontsize=13)
    ax2.set_title('Tensão RMS - Barra 820', fontsize=15, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    #plt.show()

    print("✅ Gráficos gerados com sucesso!")

else:
    print("⚠️ Execute a célula de análise (arquivo_teste) primeiro para ter dados disponíveis!")


#%%
def criar_tabela_comparativa_completa(casos_dict, salvar_como=None):
    """
    Cria tabela comparativa entre todos os casos.

    casos_dict = {
        'T2F_sem_terra_SR': proc,
        'T2F_com_terra_SR': proc,
        'MRT_sem_terra_SR': proc,
        'MRT_com_terra_SR': proc,
        ...
    }
    """
    dados_tabela = []

    for nome_caso, proc in casos_dict.items():
        stats = proc.estatisticas()
        nome_lower = nome_caso.lower()

        eh_sem_terra = ('sem_terra' in nome_lower) or ('semterra' in nome_lower)
        eh_sr = ('_sr_' in nome_lower) or nome_lower.endswith('_sr')

        tipo = []
        if 't2f' in nome_lower:
            tipo.append('T2F')
        if 'mrt' in nome_lower:
            tipo.append('MRT')

        if eh_sr:
            tipo.append('Sem Regulador')
        else:
            tipo.append('Com Regulador')

        if eh_sem_terra:
            tipo.append('Sem Terra na SE')
        else:
            tipo.append('Com Terra na SE')

        descricao = ' - '.join(tipo)

        dados_tabela.append({
            'Caso': nome_caso,
            'Descrição': descricao,
            'Barra': stats['barra'],
            'I_max_A (A)': f"{stats['i_max_A']:.1f}",
            'I_max_B (A)': f"{stats['i_max_B']:.1f}",
            'I_max_C (A)': f"{stats['i_max_C']:.1f}",
            'I_média_A (A)': f"{stats['i_media_A']:.1f}",
            'V_max_A (V)': f"{stats['v_max_A']:.0f}",
            'V_max_B (V)': f"{stats['v_max_B']:.0f}",
            'V_max_C (V)': f"{stats['v_max_C']:.0f}",
        })

    df = pd.DataFrame(dados_tabela)

    print("\n" + "="*120)
    print("📊 TABELA COMPARATIVA COMPLETA - PARA TESE")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)

    fig, ax = plt.subplots(figsize=(18, len(dados_tabela) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.18, 0.26, 0.06, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')

    plt.title('Tabela Comparativa - Análise de Sistemas T2F/MRT, SR e Sem Terra',
              fontsize=16, fontweight='bold', pad=20)

    if salvar_como:
        plt.savefig(salvar_como, dpi=600, bbox_inches='tight', facecolor='white')
        df.to_csv(salvar_como.replace('.png', '.csv'), index=False)
        df.to_excel(salvar_como.replace('.png', '.xlsx'), index=False)
        print(f"  💾 Salvo: {salvar_como}")
        print(f"  💾 CSV: {salvar_como.replace('.png', '.csv')}")
        print(f"  💾 Excel: {salvar_como.replace('.png', '.xlsx')}")

    #plt.show()

    return df

print("✅ Função criar_tabela_comparativa_completa definida!")


#%%
# 🎓 GERA FIGURAS DE ALTA QUALIDADE PARA A TESE – EXEMPLO T2F SEM TERRA

arquivo_t2f_sem_terra = (
    "C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/"
    "T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/"
    "Teste_Novo_Sem_Terra_13/Processados_HDF5/"
    "Qualificacao__R_820_-_Falta_ABC_py.mat"
)

PASTA_TESE = Path(PASTA_SAIDA) / "Capitulo_4_Figuras_Tese"
PASTA_TESE.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🎓 GERANDO FIGURAS DE ALTA QUALIDADE PARA TESE - CAPÍTULO 4")
print("="*80)

processadores_t2f_sem_terra = analisar_arquivo_completo(
    arquivo_t2f_sem_terra,
    barras_interesse=["800", "818", "820", "822", "T2F", "T2F1"],
    gerar_graficos=True
)

if processadores_t2f_sem_terra:
    print(f"\n✅ {len(processadores_t2f_sem_terra)} barras processadas com sucesso!")

    for barra, proc in processadores_t2f_sem_terra.items():
        print(f"\n📊 Gerando gráficos para Barra {barra}...")

        plotar_tensao_evento_falta(
            proc,
            f'Tensão Durante Falta ABC - Barra {barra}',
            instante_falta=0.5/3,
            salvar_como=PASTA_TESE / f'Fig_Tensao_Falta_Barra_{barra}_T2F_sem_terra_SR.png'
        )

        plotar_corrente_falta_zoom(
            proc,
            f'Corrente de Falta ABC - Barra {barra}',
            instante_falta=0.5/3,
            janela_zoom=0.05,
            salvar_como=PASTA_TESE / f'Fig_Corrente_Zoom_Barra_{barra}_T2F_sem_terra_SR.png'
        )

        plotar_clarke_profissional(
            proc,
            f'Barra {barra}',
            salvar_como=PASTA_TESE / f'Fig_Clarke_Profissional_Barra_{barra}_T2F_sem_terra_SR.png'
        )

    print("\n" + "="*80)
    print("✅ TODOS OS GRÁFICOS GERADOS COM SUCESSO!")
    print(f"📁 Localização: {PASTA_TESE}")
    print("="*80)
else:
    print("❌ Erro ao processar arquivo T2F sem terra!")


#%%
# 📊 RESUMO FINAL DA ANÁLISE

print("\n" + "="*80)
print("📊 RESUMO DA SESSÃO DE ANÁLISE")
print("="*80)

if 'processadores' in locals() and processadores:
    print(f"\n✅ Arquivo analisado com sucesso!")
    print(f"📍 Barras processadas: {list(processadores.keys())}")
    print(f"💾 Resultados salvos em: {PASTA_SAIDA}")

    print("\n📈 Resumo de Correntes Máximas:")
    for barra, proc in processadores.items():
        stats = proc.estatisticas()
        print(
            f"   Barra {barra:5s}: "
            f"A={stats['i_max_A']:6.1f}A  "
            f"B={stats['i_max_B']:6.1f}A  "
            f"C={stats['i_max_C']:6.1f}A"
        )
else:
    print("\n⚠️ Nenhuma análise foi executada ainda.")
    print("   Execute a célula de análise de arquivo específico para começar!")

print("\n" + "="*80)
print("✅ Notebook pronto para uso!")
print("="*80)


#%%

#%%


#%%
def exportar_dados_completos(processadores, pasta_saida, prefixo="analise"):
    """
    Exporta dados em múltiplos formatos para uso na tese.
    """
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("📦 EXPORTANDO DADOS EM MÚLTIPLOS FORMATOS")
    print("="*80)

    resultados = {}

    for barra, proc in processadores.items():
        print(f"\n🔹 Exportando Barra {barra}...")

        # 1. CSV - Dados brutos (tempo, tensão, corrente)
        dados_brutos = {
            'tempo_s': proc.t,
            'V_A': proc.v[:, 0],
            'V_B': proc.v[:, 1],
            'V_C': proc.v[:, 2],
            'I_A': proc.i[:, 0],
            'I_B': proc.i[:, 1],
            'I_C': proc.i[:, 2],
        }
        df_brutos = pd.DataFrame(dados_brutos)
        arquivo_csv = pasta / f'{prefixo}_Barra_{barra}_dados_brutos.csv'
        df_brutos.to_csv(arquivo_csv, index=False)
        print(f"   ✅ CSV (dados brutos): {arquivo_csv.name}")

        # 2. CSV - Dados RMS
        i_rms = proc.calcular_rms(proc.i)
        v_rms = proc.calcular_rms(proc.v)
        dados_rms = {
            'tempo_s': proc.t,
            'V_RMS_A': v_rms[:, 0],
            'V_RMS_B': v_rms[:, 1],
            'V_RMS_C': v_rms[:, 2],
            'I_RMS_A': i_rms[:, 0],
            'I_RMS_B': i_rms[:, 1],
            'I_RMS_C': i_rms[:, 2],
        }
        df_rms = pd.DataFrame(dados_rms)
        arquivo_rms = pasta / f'{prefixo}_Barra_{barra}_dados_rms.csv'
        df_rms.to_csv(arquivo_rms, index=False)
        print(f"   ✅ CSV (RMS): {arquivo_rms.name}")

#%%
def exportar_dados_completos(processadores, pasta_saida, prefixo="analise"):
    """
    Exporta dados em múltiplos formatos para uso na tese.
    """
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("📦 EXPORTANDO DADOS EM MÚLTIPLOS FORMATOS")
    print("="*80)

    resultados = {}

    for barra, proc in processadores.items():
        print(f"\n🔹 Exportando Barra {barra}...")

        # 1. CSV - Dados brutos (tempo, tensão, corrente)
        dados_brutos = {
            'tempo_s': proc.t,
            'V_A': proc.v[:, 0],
            'V_B': proc.v[:, 1],
            'V_C': proc.v[:, 2],
            'I_A': proc.i[:, 0],
            'I_B': proc.i[:, 1],
            'I_C': proc.i[:, 2],
        }
        df_brutos = pd.DataFrame(dados_brutos)
        arquivo_csv = pasta / f'{prefixo}_Barra_{barra}_dados_brutos.csv'
        df_brutos.to_csv(arquivo_csv, index=False)
        print(f"   ✅ CSV (dados brutos): {arquivo_csv.name}")

        # 2. CSV - Dados RMS
        i_rms = proc.calcular_rms(proc.i)
        v_rms = proc.calcular_rms(proc.v)
        dados_rms = {
            'tempo_s': proc.t,
            'V_RMS_A': v_rms[:, 0],
            'V_RMS_B': v_rms[:, 1],
            'V_RMS_C': v_rms[:, 2],
            'I_RMS_A': i_rms[:, 0],
            'I_RMS_B': i_rms[:, 1],
            'I_RMS_C': i_rms[:, 2],
        }
        df_rms = pd.DataFrame(dados_rms)
        arquivo_rms = pasta / f'{prefixo}_Barra_{barra}_dados_rms.csv'
        df_rms.to_csv(arquivo_rms, index=False)
        print(f"   ✅ CSV (RMS): {arquivo_rms.name}")

        # 3. Excel - Dados completos (múltiplas abas)
        arquivo_excel = pasta / f'{prefixo}_Barra_{barra}_completo.xlsx'
        with pd.ExcelWriter(arquivo_excel, engine='openpyxl') as writer:
            df_brutos.to_excel(writer, sheet_name='Dados_Brutos', index=False)
            df_rms.to_excel(writer, sheet_name='Dados_RMS', index=False)

            stats = proc.estatisticas()
            df_stats = pd.DataFrame([stats])
            df_stats.to_excel(writer, sheet_name='Estatisticas', index=False)

            alpha, beta = proc.clarke()
            df_clarke = pd.DataFrame({
                'tempo_s': proc.t,
                'alpha': alpha,
                'beta': beta,
                'magnitude': np.sqrt(alpha**2 + beta**2)
            })
            df_clarke.to_excel(writer, sheet_name='Clarke', index=False)
        print(f"   ✅ Excel (completo): {arquivo_excel.name}")

        # 4. JSON - Estatísticas
        stats_json = proc.estatisticas()
        import json
        arquivo_json = pasta / f'{prefixo}_Barra_{barra}_estatisticas.json'
        with open(arquivo_json, 'w', encoding='utf-8') as f:
            json.dump(stats_json, f, indent=4, ensure_ascii=False)
        print(f"   ✅ JSON (estatísticas): {arquivo_json.name}")

        # 5. NPZ - Numpy (para processamento posterior)
        arquivo_npz = pasta / f'{prefixo}_Barra_{barra}_numpy.npz'
        np.savez_compressed(
            arquivo_npz,
            tempo=proc.t,
            tensao=proc.v,
            corrente=proc.i,
            tensao_rms=v_rms,
            corrente_rms=i_rms,
            alpha=alpha,
            beta=beta
        )
        print(f"   ✅ NPZ (numpy): {arquivo_npz.name}")

        # 6. LaTeX - Tabela para incluir na tese
        arquivo_tex = pasta / f'{prefixo}_Barra_{barra}_tabela.tex'
        df_stats_formatado = pd.DataFrame([{
            'Barra': barra,
            'I_{max,A} (A)': f"{stats_json['i_max_A']:.2f}",
            'I_{max,B} (A)': f"{stats_json['i_max_B']:.2f}",
            'I_{max,C} (A)': f"{stats_json['i_max_C']:.2f}",
            'V_{max,A} (V)': f"{stats_json['v_max_A']:.0f}",
            'V_{max,B} (V)': f"{stats_json['v_max_B']:.0f}",
            'V_{max,C} (V)': f"{stats_json['v_max_C']:.0f}",
        }])
        with open(arquivo_tex, 'w', encoding='utf-8') as f:
            f.write(df_stats_formatado.to_latex(index=False, escape=False))
        print(f"   ✅ LaTeX (tabela): {arquivo_tex.name}")

        resultados[barra] = {
            'csv_brutos': arquivo_csv,
            'csv_rms': arquivo_rms,
            'excel': arquivo_excel,
            'json': arquivo_json,
            'npz': arquivo_npz,
            'latex': arquivo_tex
        }

    print("\n" + "="*80)
    print("✅ EXPORTAÇÃO CONCLUÍDA!")
    print(f"📁 Todos os arquivos em: {pasta}")
    print("="*80)

    return resultados

print("✅ Função exportar_dados_completos definida!")


#%%
def salvar_grafico_multiplos_formatos(fig, nome_base, pasta_saida):
    """
    Salva gráfico em PNG, PDF, SVG e EPS.
    """
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)

    formatos = {
        'png': {'dpi': 600, 'desc': 'PNG (alta resolução)'},
        'pdf': {'dpi': 600, 'desc': 'PDF (vetorial)'},
        'svg': {'dpi': None, 'desc': 'SVG (vetorial editável)'},
        'eps': {'dpi': 600, 'desc': 'EPS (publicação)'},
    }

    arquivos_salvos = []

    for formato, config in formatos.items():
        arquivo = pasta / f"{nome_base}.{formato}"
        if config['dpi']:
            fig.savefig(arquivo, format=formato, dpi=config['dpi'],
                        bbox_inches='tight', facecolor='white')
        else:
            fig.savefig(arquivo, format=formato,
                        bbox_inches='tight', facecolor='white')
        arquivos_salvos.append(arquivo)
        print(f"   ✅ {config['desc']}: {arquivo.name}")

    return arquivos_salvos

print("✅ Função salvar_grafico_multiplos_formatos definida!")


#%%
def exportar_analise_fft(proc, barra, pasta_saida, prefixo="fft"):
    """
    Exporta análise completa de FFT e harmônicos.
    """
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)

    print(f"\n🔊 Analisando FFT - Barra {barra}...")

    dados_fft = {'frequencia_Hz': None}
    for fase_idx, fase_nome in enumerate(['A', 'B', 'C']):
        freqs, mag = proc.fft_espectro(fase_idx, n_cycles=2.0)
        if dados_fft['frequencia_Hz'] is None:
            dados_fft['frequencia_Hz'] = freqs
        dados_fft[f'magnitude_fase_{fase_nome}'] = mag

    df_fft = pd.DataFrame(dados_fft)
    df_fft = df_fft[df_fft['frequencia_Hz'] <= 2000]

    arquivo_csv = pasta / f'{prefixo}_Barra_{barra}_espectro_completo.csv'
    df_fft.to_csv(arquivo_csv, index=False)
    print(f"   ✅ CSV (espectro): {arquivo_csv.name}")

    fig, ax = plt.subplots(figsize=(12, 6))
    for fase_nome in ['A', 'B', 'C']:
        ax.stem(
            df_fft['frequencia_Hz'],
            df_fft[f'magnitude_fase_{fase_nome}'],
            linefmt='-', markerfmt=' ', basefmt=' ',
            label=f'Fase {fase_nome}'
        )
    ax.set_xlim(0, 1000)
    ax.set_xlabel('Frequência (Hz)', fontweight='bold')
    ax.set_ylabel('Magnitude (A)', fontweight='bold')
    ax.set_title(f'Espectro de Corrente - Barra {barra}', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    arquivos_grafico = salvar_grafico_multiplos_formatos(
        fig,
        nome_base=f'{prefixo}_Barra_{barra}_espectro',
        pasta_saida=pasta
    )
    plt.close(fig)

    return {
        'csv_espectro': arquivo_csv,
        'figuras': arquivos_grafico
    }

print("✅ Função exportar_analise_fft definida!")


#%%
def gerar_relatorio_html(processadores, pasta_saida, nome_relatorio="relatorio_analise.html"):
    """
    Gera um relatório HTML simples com estatísticas por barra.
    """
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)

    caminho_html = pasta / nome_relatorio

    linhas = []
    linhas.append("<html><head><meta charset='utf-8'><title>Relatório de Análise</title></head><body>")
    linhas.append("<h1>Relatório de Análise de Faltas</h1>")

    for barra, proc in processadores.items():
        stats = proc.estatisticas()
        linhas.append(f"<h2>Barra {barra}</h2>")
        linhas.append("<ul>")
        linhas.append(f"<li>I_max_A: {stats['i_max_A']:.2f} A</li>")
        linhas.append(f"<li>I_max_B: {stats['i_max_B']:.2f} A</li>")
        linhas.append(f"<li>I_max_C: {stats['i_max_C']:.2f} A</li>")
        linhas.append(f"<li>V_max_A: {stats['v_max_A']:.0f} V</li>")
        linhas.append(f"<li>V_max_B: {stats['v_max_B']:.0f} V</li>")
        linhas.append(f"<li>V_max_C: {stats['v_max_C']:.0f} V</li>")
        linhas.append("</ul>")

    linhas.append("</body></html>")

    with open(caminho_html, 'w', encoding='utf-8') as f:
        f.write("\n".join(linhas))

    print(f"✅ Relatório HTML gerado: {caminho_html}")
    return caminho_html
#%%
# EXEMPLO: exportar dados e FFT para o caso T2F sem terra, sem regulador (_SR_)

if 'processadores_t2f_sem_terra' in locals() and processadores_t2f_sem_terra:
    pasta_export = PASTA_TESE / "Exportacoes_T2F_sem_terra_SR"
    resultados_export = exportar_dados_completos(
        processadores_t2f_sem_terra,
        pasta_saida=pasta_export,
        prefixo="T2F_sem_terra_SR"
    )

    for barra, proc in processadores_t2f_sem_terra.items():
        exportar_analise_fft(
            proc,
            barra=barra,
            pasta_saida=pasta_export,
            prefixo="fft_T2F_sem_terra_SR"
        )

    gerar_relatorio_html(
        processadores_t2f_sem_terra,
        pasta_saida=pasta_export,
        nome_relatorio="Relatorio_T2F_sem_terra_SR.html"
    )
else:
    print("⚠️ Processadores T2F sem terra não carregados. Execute a análise T2F primeiro.")

#%%
# 🚀 RODAR ANÁLISE PARA TODOS OS ARQUIVOS DA PASTA_DADOS

resultados_lote = analisar_multiplos_arquivos(
    PASTA_DADOS,
    padrao="*.mat",
    barras_interesse=["800", "818", "820", "822", "T2F", "T2F1"]
)