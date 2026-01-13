# %%
# ==============================================================================
# 1. IMPORTS E CONFIGURAÇÕES GERAIS
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import scipy.io as sio
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import warnings
import matplotlib as mpl

# Suprimir avisos
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÕES DE ESTILO (TESE - ALTA QUALIDADE) ---
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 120,
    'savefig.dpi': 600,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Cores
CORES = {'A': '#0066CC', 'B': '#CC0000', 'C': '#009900'}  # Cores das Fases
CORES_COMP = {'MRN': '#1f77b4', 'T2F': '#ff7f0e'}  # Cores Topologia

print("✅ Bibliotecas e Estilos Configurados.")


# ==============================================================================
# 2. CLASSE PROCESSADORA (ENGINE)
# ==============================================================================
class ProcessadorSinais:
    def __init__(self, t, V, I, freq=60, barra_nome=""):
        self.t = np.array(t).flatten()
        self.v = np.array(V)
        self.i = np.array(I)
        self.freq = freq
        self.barra_nome = barra_nome

        # Taxa de amostragem
        if len(self.t) > 1:
            self.dt = float(self.t[1] - self.t[0])
            self.fs = 1.0 / self.dt
        else:
            self.dt = 1 / 60 / 256
            self.fs = 60 * 256
        self.samples_per_cycle = int(self.fs / self.freq)

    def calcular_rms(self, sinal):
        """RMS móvel."""
        window = max(1, self.samples_per_cycle)
        return np.sqrt(uniform_filter1d(sinal ** 2, size=window, axis=0))

    def clarke(self):
        """Transformada de Clarke (Alpha, Beta)."""
        a, b, c = self.i[:, 0], self.i[:, 1], self.i[:, 2]
        alpha = (2 * a - b - c) / 3.0
        beta = (b - c) / np.sqrt(3.0)
        return alpha, beta

    def componentes_simetricas(self, idx):
        """Componentes simétricas num instante específico."""
        ia, ib, ic = self.i[idx, 0], self.i[idx, 1], self.i[idx, 2]
        a = np.exp(1j * 2 * np.pi / 3)
        I0 = np.abs((ia + ib + ic) / 3.0)
        I1 = np.abs((ia + a * ib + a ** 2 * ic) / 3.0)
        I2 = np.abs((ia + a ** 2 * ib + a * ic) / 3.0)
        return I0, I1, I2

    def get_sym_components_sequence(self, t_instante):
        """Para análise comparativa (retorna array de magnitudes)."""
        idx = np.searchsorted(self.t, t_instante)
        return self.componentes_simetricas(idx)

    def fft_espectro(self, fase_idx, n_cycles=2.0):
        """FFT para harmônicas."""
        n = int(max(16, round(self.samples_per_cycle * n_cycles)))
        idx_fim = len(self.v) - 1
        idx_inicio = max(0, idx_fim - n + 1)

        seg = self.v[idx_inicio:idx_fim + 1, fase_idx]  # FFT DA TENSÃO
        seg = seg - np.mean(seg)
        w = np.hanning(len(seg))
        X = np.fft.rfft(seg * w)
        freqs = np.fft.rfftfreq(len(seg), d=self.dt)
        mag = (2.0 / np.sum(w)) * np.abs(X)
        return freqs, mag

    def get_thd_harmonics(self, t_inicio, t_fim):
        """Calcula V1, V3 e THD."""
        freqs, mag = self.fft_espectro(0, n_cycles=3)  # Fase A
        idx_60 = np.argmin(np.abs(freqs - 60))
        idx_180 = np.argmin(np.abs(freqs - 180))

        v1 = mag[idx_60] if idx_60 < len(mag) else 0
        v3 = mag[idx_180] if idx_180 < len(mag) else 0

        # THD simples (soma dos quadrados das harmonicas / fundamental)
        harmonics_sum = np.sum(mag[idx_60 + 1:] ** 2)
        thd = (np.sqrt(harmonics_sum) / v1) * 100 if v1 > 0 else 0
        return v1, v3, thd

    def estatisticas(self):
        i_rms = self.calcular_rms(self.i)
        v_rms = self.calcular_rms(self.v)
        return {
            'barra': self.barra_nome,
            'i_max_A': np.max(np.abs(i_rms[:, 0])),
            'i_max_B': np.max(np.abs(i_rms[:, 1])),
            'i_max_C': np.max(np.abs(i_rms[:, 2])),
            'v_max_A': np.max(np.abs(v_rms[:, 0])),
            'v_max_B': np.max(np.abs(v_rms[:, 1])),
            'v_max_C': np.max(np.abs(v_rms[:, 2])),
        }


# ==============================================================================
# 3. FUNÇÕES DE CARREGAMENTO ROBUSTO
# ==============================================================================
def _garantir_3_fases(arr):
    arr = np.array(arr)
    if arr.ndim == 1: return np.column_stack([arr, arr, arr])
    if arr.shape[0] == 3 and arr.shape[1] > 3: return arr.T
    return arr


def extrair_dados_barra(dados, barra):
    chaves_v = [f'V_{barra}_raw', f'V_{barra}', f'v_{barra}_raw']
    chaves_i = [f'I_{barra}_raw', f'I_{barra}', f'i_{barra}_raw']
    t_raw = dados.get('t', dados.get('time', dados.get('tempo', None)))

    if t_raw is None: return None, None, None
    t = np.array(t_raw).flatten()

    V, I = None, None
    for k in chaves_v:
        if k in dados: V = dados[k]; break
    for k in chaves_i:
        if k in dados: I = dados[k]; break

    if V is None or I is None: return None, None, None

    V = _garantir_3_fases(V)
    I = _garantir_3_fases(I)
    L = min(len(t), len(V), len(I))
    return t[:L], V[:L], I[:L]


def carregar_arquivo_completo(caminho):
    """Lê todas as barras de um arquivo .mat."""
    dados = {}
    try:
        # Tenta ler HDF5 ou MAT
        try:
            with h5py.File(caminho, 'r') as f:
                for k in f.keys():
                    if not k.startswith('__'):
                        d = f[k][()]
                        if isinstance(d, np.ndarray) and d.shape[0] == 3 and d.shape[1] > 3: d = d.T
                        dados[k] = d
        except:
            dados = sio.loadmat(caminho, squeeze_me=False)

        # Identificar barras
        barras = set()
        for k in dados.keys():
            if '_raw' in k and 'V_' in k:
                partes = k.split('_')
                if len(partes) >= 2: barras.add(partes[1])

        processadores = {}
        for b in sorted(list(barras)):
            t, V, I = extrair_dados_barra(dados, b)
            if t is not None:
                processadores[b] = ProcessadorSinais(t, V, I, barra_nome=b)

        return processadores

    except Exception as e:
        print(f"❌ Erro lendo {caminho}: {e}")
        return None


# ==============================================================================
# 4. FUNÇÕES DE PLOTAGEM INDIVIDUAL (ESTILO CÓDIGO ANTIGO)
# ==============================================================================
def plotar_individuais(processadores, pasta_saida, nome_base):
    """Gera os gráficos detalhados para cada barra deste arquivo."""
    pasta_arq = Path(pasta_saida) / nome_base
    pasta_arq.mkdir(parents=True, exist_ok=True)

    # 1. RMS Multiplas Barras
    n = len(processadores)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n), sharex=True)
    if n == 1: axes = axes.reshape(1, -1)

    for idx, (b, proc) in enumerate(processadores.items()):
        vrms, irms = proc.calcular_rms(proc.v), proc.calcular_rms(proc.i)
        # Tensão
        axes[idx, 0].plot(proc.t, vrms[:, 0], c=CORES['A'], label='A');
        axes[idx, 0].plot(proc.t, vrms[:, 1], c=CORES['B'], label='B');
        axes[idx, 0].plot(proc.t, vrms[:, 2], c=CORES['C'], label='C')
        axes[idx, 0].set_ylabel(f'V RMS - {b}', fontweight='bold');
        axes[idx, 0].grid(True, alpha=0.3)
        # Corrente
        axes[idx, 1].plot(proc.t, irms[:, 0], c=CORES['A'], label='A');
        axes[idx, 1].plot(proc.t, irms[:, 1], c=CORES['B'], label='B');
        axes[idx, 1].plot(proc.t, irms[:, 2], c=CORES['C'], label='C')
        axes[idx, 1].set_ylabel(f'I RMS - {b}', fontweight='bold');
        axes[idx, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title(f'Tensões RMS - {nome_base}');
    axes[0, 1].set_title(f'Correntes RMS - {nome_base}')
    plt.tight_layout();
    plt.savefig(pasta_arq / f'RMS_Geral.png');
    plt.close()

    # 2. Detalhes por Barra (Clarke e Zoom)
    for b, proc in processadores.items():
        # Clarke
        alpha, beta = proc.clarke()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.plot(alpha, beta, alpha=0.4);
        ax1.plot(alpha[-300:], beta[-300:], c='orange', lw=2);
        ax1.axis('equal')
        ax1.set_title(f'Clarke αβ - {b}');
        ax1.grid(True)
        mag = np.sqrt(alpha ** 2 + beta ** 2)
        ax2.plot(proc.t, mag, c='green');
        ax2.set_title('Magnitude Instantânea');
        ax2.grid(True)
        plt.tight_layout();
        plt.savefig(pasta_arq / f'Clarke_{b}.png');
        plt.close()

        # Zoom Corrente
        irms = proc.calcular_rms(proc.i)
        t_falta = 0.5 / 3  # Estimativa
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(proc.t, irms[:, 0], c=CORES['A']);
        ax.plot(proc.t, irms[:, 1], c=CORES['B']);
        ax.plot(proc.t, irms[:, 2], c=CORES['C'])
        ax.set_title(f'Corrente RMS - {b}');
        ax.grid(True)
        plt.savefig(pasta_arq / f'Corrente_{b}.png');
        plt.close()


# ==============================================================================
# 5. FUNÇÕES DE PLOTAGEM COMPARATIVA (ESTILO NOVO)
# ==============================================================================

def plot_secao_1_aterramento(BD, barra_alvo, salvar_dir):
    print(f"   -> Gerando Seção 1 (Aterramento) para {barra_alvo}")
    for topo in ['MRN', 'T2F']:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        fig.suptitle(f'Impacto Aterramento - {topo} - {barra_alvo}', fontsize=16)
        configs = [(0, 0, 'ComReg', 'ComTerra'), (0, 1, 'ComReg', 'SemTerra'),
                   (1, 0, 'SemReg', 'ComTerra'), (1, 1, 'SemReg', 'SemTerra')]

        for r, c, reg, terra in configs:
            ax = axes[r, c]
            proc = BD[topo][reg][terra].get(barra_alvo)
            if proc:
                rms = proc.calcular_rms(proc.v)
                ax.plot(proc.t, rms[:, 0], c=CORES['A']);
                ax.plot(proc.t, rms[:, 1], c=CORES['B']);
                ax.plot(proc.t, rms[:, 2], c=CORES['C'])
            ax.set_title(f'{reg} / {terra}');
            ax.grid(True, alpha=0.3)
        plt.tight_layout();
        plt.savefig(Path(salvar_dir) / f'Sec1_{topo}_{barra_alvo}.png');
        plt.close()


def plot_secao_2_harmonicos(BD, barra_alvo, salvar_dir):
    print(f"   -> Gerando Seção 2 (Harmônicas) para {barra_alvo}")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'FFT Tensão (Zoom) - {barra_alvo}', fontsize=16)
    configs = [('MRN', 'ComReg', axes[0, 0]), ('T2F', 'ComReg', axes[0, 1]),
               ('MRN', 'SemReg', axes[1, 0]), ('T2F', 'SemReg', axes[1, 1])]

    dados_csv = []
    for topo, reg, ax in configs:
        for terra, ls in [('ComTerra', '-'), ('SemTerra', '--')]:
            proc = BD[topo][reg][terra].get(barra_alvo)
            if proc:
                f, m = proc.fft_espectro(0, n_cycles=3)
                mask = f <= 360
                ax.plot(f[mask], m[mask], ls=ls, label=terra)
                v1, v3, thd = proc.get_thd_harmonics(0, 100)  # Dummy times
                dados_csv.append({'Topo': topo, 'Reg': reg, 'Terra': terra, 'V1': v1, 'V3': v3, 'THD': thd})
        ax.set_title(f'{topo} - {reg}');
        ax.legend();
        ax.grid(True, alpha=0.3)

    plt.tight_layout();
    plt.savefig(Path(salvar_dir) / f'Sec2_FFT_{barra_alvo}.png');
    plt.close()
    pd.DataFrame(dados_csv).to_csv(Path(salvar_dir) / f'Sec2_Tabela_{barra_alvo}.csv', index=False)


def plot_secao_3_comparativo(BD, barra_prot, barra_qual, salvar_dir):
    print(f"   -> Gerando Seção 3 (Comparativo)")
    resumo = []
    for topo in ['MRN', 'T2F']:
        for reg in ['ComReg', 'SemReg']:
            proc_p = BD[topo][reg]['ComTerra'].get(barra_prot)
            proc_q = BD[topo][reg]['ComTerra'].get(barra_qual)
            imax = np.max(proc_p.calcular_rms(proc_p.i)) if proc_p else 0

            deseq = 0
            if proc_q:
                seq = proc_q.get_sym_components_sequence(proc_q.t[-10])  # Fim da falta
                deseq = (seq[2] / seq[1]) * 100 if seq[1] > 0 else 0
            resumo.append({'Label': f'{topo}\n{reg}', 'Topo': topo, 'Imax': imax, 'V2/V1': deseq})

    df = pd.DataFrame(resumo)
    if df.empty: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cols = [CORES_COMP[r['Topo']] for _, r in df.iterrows()]
    ax1.bar(df['Label'], df['Imax'], color=cols);
    ax1.set_title(f'I Falta Máx - {barra_prot}')
    ax2.bar(df['Label'], df['V2/V1'], color=cols);
    ax2.set_title(f'Desequilíbrio V2/V1 - {barra_qual}')
    plt.tight_layout();
    plt.savefig(Path(salvar_dir) / 'Sec3_Comparativo.png');
    plt.close()


def plot_secao_4_perfil(BD, salvar_dir):
    print(f"   -> Gerando Seção 4 (Perfil Tensão)")
    fig, ax = plt.subplots(figsize=(10, 6))
    barras = ['800', '816', '820', '822']
    x = range(len(barras))

    for topo in ['MRN', 'T2F']:
        for reg, ls in [('ComReg', '-'), ('SemReg', '--')]:
            vals = []
            for b in barras:
                proc = BD[topo][reg]['ComTerra'].get(b)
                if proc:
                    rms = proc.calcular_rms(proc.v)
                    vals.append(np.mean(rms[:100, :]))  # Pre-falta
                else:
                    vals.append(None)
            if None not in vals:
                ax.plot(x, vals, ls=ls, marker='o', label=f'{topo} {reg}')

    ax.set_xticks(x);
    ax.set_xticklabels(barras);
    ax.legend();
    ax.grid(True, alpha=0.3)
    ax.set_title('Perfil de Tensão - Regime Permanente')
    plt.savefig(Path(salvar_dir) / 'Sec4_Perfil.png');
    plt.close()


# ==============================================================================
# 6. EXECUÇÃO PRINCIPAL
# ==============================================================================

# 🔧 CONFIGURE AQUI
PASTA_RAIZ = r"C:/Users/Leonardo Felipe/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/"
PASTA_SAIDA = "./Resultados_Completo_v14"

# Estrutura Banco de Dados para Comparação
BD = {'MRN': {'ComReg': {'ComTerra': {}, 'SemTerra': {}}, 'SemReg': {'ComTerra': {}, 'SemTerra': {}}},
      'T2F': {'ComReg': {'ComTerra': {}, 'SemTerra': {}}, 'SemReg': {'ComTerra': {}, 'SemTerra': {}}}}


def main():
    arquivos = list(Path(PASTA_RAIZ).glob("*.mat"))
    print(f"🚀 INICIANDO PROCESSAMENTO DE {len(arquivos)} ARQUIVOS\n")
    Path(PASTA_SAIDA).mkdir(parents=True, exist_ok=True)

    # 1. LOOP DE LEITURA E ANÁLISE INDIVIDUAL
    for arq in arquivos:
        print(f"📄 Processando: {arq.name} ...")

        # A. Carrega TODAS as barras deste arquivo
        procs_arquivo = carregar_arquivo_completo(arq)

        if procs_arquivo:
            # B. Gera Relatórios Individuais (Estilo Código Antigo)
            plotar_individuais(procs_arquivo, PASTA_SAIDA, arq.stem)

            # C. Classifica e Salva no BD para Comparação (Estilo Código Novo)
            nome = arq.name.lower()
            topo = 'T2F' if 't2f' in nome else 'MRN'
            reg = 'SemReg' if ('_sr_' in nome or 'sem_reg' in nome or 'semreg' in nome) else 'ComReg'
            terra = 'SemTerra' if ('sem_terra' in nome or 'semterra' in nome) else 'ComTerra'

            # Salva cada barra no local correto do BD
            for barra, proc in procs_arquivo.items():
                BD[topo][reg][terra][barra] = proc
        else:
            print(f"   ⚠️ Falha ao ler {arq.name}")

    print("\n✅ ANÁLISE INDIVIDUAL CONCLUÍDA. INICIANDO ANÁLISE COMPARATIVA...\n")

    # 2. ANÁLISE COMPARATIVA (SEÇÕES 1-4)
    pasta_comparativa = Path(PASTA_SAIDA) / "Comparativo_Final_Tese"
    pasta_comparativa.mkdir(parents=True, exist_ok=True)

    try:
        plot_secao_1_aterramento(BD, '820', pasta_comparativa)
        plot_secao_1_aterramento(BD, '822', pasta_comparativa)

        plot_secao_2_harmonicos(BD, '820', pasta_comparativa)

        plot_secao_3_comparativo(BD, '816', '822', pasta_comparativa)

        plot_secao_4_perfil(BD, pasta_comparativa)
        print("\n✅ TODAS AS ANÁLISES CONCLUÍDAS COM SUCESSO!")
        print(f"📂 Resultados salvos em: {PASTA_SAIDA}")

    except Exception as e:
        print(f"❌ Erro na etapa comparativa: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
# %%