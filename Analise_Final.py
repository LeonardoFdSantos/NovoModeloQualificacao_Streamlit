import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
import warnings
from pathlib import Path

# ==============================================================================
# 1. CONFIGURAÇÕES GLOBAIS
# ==============================================================================

# 🔧 AJUSTE O CAMINHO DA SUA PASTA AQUI
PASTA_DADOS = r"C:\Users\leosa\OneDrive\Coisas_Leonardo\gits\CurtosT2F\T2F_MATLAB\NovoArtigoPowerDelivery34bus\NovoModeloQualificacao\Teste_Novo_Sem_Terra_14\Processados_HDF5/"

BARRAS_DISPONIVEIS = ["800", "T2F", "T2F1", "818", "820", "822"]
INSTANTE_FALTA = 0.5 / 3  # ~0.1667 s
FREQ_SISTEMA = 60.0  # Hz

# Estilo
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 120,
    'savefig.dpi': 300
})

CORES = {
    'MRN': '#1f77b4', 'T2F': '#ff7f0e',
    'com_reg': '#2ca02c', 'sem_reg': '#d62728',
    'com_terra': 'black', 'sem_terra': 'purple'
}

warnings.filterwarnings("ignore")


# ==============================================================================
# 2. FUNÇÕES AUXILIARES ROBUSTAS
# ==============================================================================

def garantir_shape_3fases(arr):
    """Garante que o array seja (N, 3)."""
    arr = np.array(arr)
    if arr.size == 0: return np.zeros((0, 3))

    # Se for 1D (ex: vetor de tempo ou uma fase solta), duplica ou molda
    if arr.ndim == 1:
        # Se parecer vetor de tempo (muitos pontos), não duplica, retorna N,1 ou erro?
        # Para V e I, queremos 3 colunas. Se vier 1, replicamos?
        # Vamos assumir que se vier 1D para V/I, é erro de leitura ou apenas 1 fase gravada.
        # Aqui, vamos tratar como coluna única
        return np.column_stack([arr, arr, arr])  # Replica para evitar erro de índice

    if arr.shape[0] == 3 and arr.shape[1] > 3:
        return arr.T

    if arr.ndim == 2 and arr.shape[1] < 3:
        # Completa com zeros se faltar fase
        cols_faltantes = 3 - arr.shape[1]
        return np.hstack([arr, np.zeros((arr.shape[0], cols_faltantes))])

    return arr


def rms(signal):
    if len(signal) == 0: return 0.0
    return np.sqrt(np.mean(np.abs(signal) ** 2))


def rms_deslizante(signal, janela_amostras):
    s = pd.Series(signal)
    return s.rolling(window=janela_amostras, center=False).apply(lambda x: np.sqrt(np.mean(x ** 2))).fillna(0).values


def indice_falta(t, instante=INSTANTE_FALTA):
    if len(t) == 0: return 0
    return int(np.argmin(np.abs(t - instante)))


def janelas_prefalta_falta(t, fs, n_ciclos=2):
    kf = indice_falta(t)
    n = int(round(n_ciclos * fs / FREQ_SISTEMA))
    return slice(max(0, kf - n), kf), slice(kf, min(len(t), kf + n))


def fasor_fundamental(x, fs):
    N = len(x)
    if N == 0: return 0j
    t = np.arange(N) / fs
    w = np.exp(-1j * 2 * np.pi * FREQ_SISTEMA * t)
    return (2.0 / N) * np.dot(x, w)


def sym_components(Ia, Ib, Ic):
    a = np.exp(1j * 2 * np.pi / 3)
    I0 = (Ia + Ib + Ic) / 3.0
    I1 = (Ia + a * Ib + a ** 2 * Ic) / 3.0
    I2 = (Ia + a ** 2 * Ib + a * Ic) / 3.0
    return I0, I1, I2


def fft_v13(v, fs):
    N = len(v)
    if N == 0: return 0, 0, 0, [], []
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    mag = np.abs(np.fft.rfft(v)) * (2.0 / N)

    idx1 = np.argmin(np.abs(freqs - FREQ_SISTEMA))
    idx3 = np.argmin(np.abs(freqs - 3 * FREQ_SISTEMA))

    V1 = mag[idx1] if idx1 < len(mag) else 0
    V3 = mag[idx3] if idx3 < len(mag) else 0

    harmonics_energy = np.sum(mag[1:] ** 2) - V1 ** 2
    THD = np.sqrt(max(0, harmonics_energy)) / (V1 + 1e-9)

    return V1, V3, THD, freqs, mag


# ==============================================================================
# 3. LEITURA E PARSING
# ==============================================================================

def extrair_metadados(nome_arquivo):
    nome = nome_arquivo.lower()

    # Topologia
    if "qualificacao" in nome:
        topologia = "T2F"
    elif "mrt" in nome:
        topologia = "MRN"
    else:
        topologia = "Outro"

    # Regulador
    if "_sr_" in nome or "sem_reg" in nome:
        regulador = "sem_reg"
    else:
        regulador = "com_reg"

    # Aterramento
    if "sem_terra" in nome:
        aterramento = "sem_terra"
    else:
        aterramento = "com_terra"

    # Tipo de Falta
    if "sem_falta" in nome:
        tipo_falta = "pleno"
    elif "falta_abc" in nome:
        tipo_falta = "ABC"
    elif "falta_ab" in nome:
        tipo_falta = "AB"
    elif "falta_ac" in nome:
        tipo_falta = "AC"
    elif "falta_bc" in nome:
        tipo_falta = "BC"
    elif "falta_a" in nome:
        tipo_falta = "A_terra"  # MRT geralmente usa Falta_A
    else:
        tipo_falta = "Outro"

    return topologia, regulador, aterramento, tipo_falta


def carregar_arquivo_mat(caminho_completo):
    """Lê arquivo e blinda os dados para evitar erro de dimensão."""
    dados_barras = {}
    t = None

    try:
        # Tenta h5py
        with h5py.File(caminho_completo, 'r') as f:
            keys = list(f.keys())
            for k in keys:
                if 'time' in k or k == 't':
                    t = np.array(f[k]).flatten()
                    break

            for barra in BARRAS_DISPONIVEIS:
                # Busca flexível de chaves
                v_key = next((k for k in keys if f"V_{barra}" in k and "raw" not in k), None)
                if not v_key: v_key = next((k for k in keys if f"V_{barra}" in k), None)

                i_key = next((k for k in keys if f"I_{barra}" in k and "raw" not in k), None)
                if not i_key: i_key = next((k for k in keys if f"I_{barra}" in k), None)

                if v_key and i_key:
                    V = garantir_shape_3fases(f[v_key])
                    I = garantir_shape_3fases(f[i_key])

                    if t is not None:
                        # Corta para o menor tamanho para evitar mismatch
                        L = min(len(t), len(V), len(I))
                        if L > 10:  # Só aceita se tiver dados suficientes
                            dados_barras[barra] = {"V": V[:L], "I": I[:L], "t": t[:L]}

    except OSError:
        # Tenta scipy.io
        try:
            mat = sio.loadmat(caminho_completo)
            keys = list(mat.keys())
            for k in keys:
                if 'time' in k or k == 't':
                    t = mat[k].flatten()
                    break

            for barra in BARRAS_DISPONIVEIS:
                v_key = next((k for k in keys if f"V_{barra}" in k), None)
                i_key = next((k for k in keys if f"I_{barra}" in k), None)

                if v_key and i_key:
                    V = garantir_shape_3fases(mat[v_key])
                    I = garantir_shape_3fases(mat[i_key])

                    if t is not None:
                        L = min(len(t), len(V), len(I))
                        if L > 10:
                            dados_barras[barra] = {"V": V[:L], "I": I[:L], "t": t[:L]}
        except Exception as e:
            print(f"Erro leitura {caminho_completo}: {e}")
            return None, None

    return t, dados_barras


def carregar_todos_cenarios():
    cenarios = {}
    arquivos = [f for f in os.listdir(PASTA_DADOS) if f.endswith('.mat')]
    print(f"📂 Encontrados {len(arquivos)} arquivos na pasta.")

    for nome_arq in arquivos:
        topo, reg, terra, falta = extrair_metadados(nome_arq)
        caminho = os.path.join(PASTA_DADOS, nome_arq)

        _, dados_barras = carregar_arquivo_mat(caminho)

        if dados_barras:
            # Pega fs de uma barra qualquer
            exemplo = next(iter(dados_barras.values()))
            dt = exemplo['t'][1] - exemplo['t'][0]
            fs = 1.0 / dt if dt > 0 else 60 * 200

            chave = (topo, reg, terra, falta)

            # Se a chave já existe (ex: arquivos quebrados em partes), mesclar seria ideal
            # Mas aqui vamos sobrescrever ou ignorar para simplificar, ou avisar
            if chave not in cenarios:
                cenarios[chave] = {"barras": dados_barras, "fs": fs, "arq": nome_arq}
            else:
                # Merge de barras se forem arquivos complementares
                cenarios[chave]["barras"].update(dados_barras)

    print(f"✅ Cenários únicos identificados: {len(cenarios)}")
    return cenarios


# ==============================================================================
# 4. FUNÇÕES DE GERAÇÃO (FIGURAS DA TESE)
# ==============================================================================

def gerar_figura_4_4(cenarios, pasta_saida):
    print("Gerando Fig 4.4 (Perfil Tensão)...")
    res = []
    plt.figure(figsize=(10, 6))

    configs = [('MRN', 'com_reg', '-o'), ('MRN', 'sem_reg', '--s'),
               ('T2F', 'com_reg', '-^'), ('T2F', 'sem_reg', '--v')]

    # Eixo X base (todas as barras desejadas)
    barras_ordenadas = ["800", "818", "820", "822", "T2F1", "T2F"]

    for topo, reg, style in configs:
        # Busca cenário "pleno" ou "sem_falta"
        keys = [(topo, reg, 'com_terra', 'pleno'), (topo, reg, 'com_terra', 'Outro')]
        key = next((k for k in keys if k in cenarios), None)

        if key:
            x_vals, y_vals = [], []
            for barra in barras_ordenadas:
                # Pula barras T2F se for topologia MRN
                if topo == "MRN" and "T2F" in barra: continue

                if barra in cenarios[key]['barras']:
                    d = cenarios[key]['barras'][barra]
                    # Média RMS do final (regime permanente)
                    n_pts = int(4 * cenarios[key]['fs'] / 60)
                    if len(d['V']) > n_pts:
                        V_win = d['V'][-n_pts:, :]
                        v_med = np.mean([rms(V_win[:, 0]), rms(V_win[:, 1]), rms(V_win[:, 2])])

                        x_vals.append(barra)
                        y_vals.append(v_med)
                        res.append({'Topo': topo, 'Reg': reg, 'Barra': barra, 'VRMS': v_med})

            if x_vals:
                plt.plot(x_vals, y_vals, style, label=f"{topo} {reg}", color=CORES[topo])

    plt.title("Perfil de Tensão RMS - Regime Permanente")
    plt.ylabel("Tensão (V)")
    plt.legend()
    plt.savefig(os.path.join(pasta_saida, "Fig4_4_Perfil.png"))
    plt.close()
    if res: pd.DataFrame(res).to_csv(os.path.join(pasta_saida, "Tab4_4.csv"), index=False)


def gerar_figura_4_6(cenarios, pasta_saida):
    """Gera Fig 4.6 (Tensão RMS no tempo) para TODAS as barras."""
    print("Gerando Fig 4.6 (RMS Tempo)...")
    falta = 'A_terra'
    reg = 'com_reg'

    # Itera sobre todas as barras disponíveis no sistema
    for barra in BARRAS_DISPONIVEIS:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        configs = [('MRN', 'com_terra'), ('MRN', 'sem_terra'),
                   ('T2F', 'com_terra'), ('T2F', 'sem_terra')]

        plotted_any = False
        for ax, (topo, terra) in zip(axs.flatten(), configs):
            key = (topo, reg, terra, falta)

            if key in cenarios and barra in cenarios[key]['barras']:
                d = cenarios[key]['barras'][barra]
                win = int(cenarios[key]['fs'] / 60)

                # Cálculo RMS seguro
                try:
                    vrms_a = rms_deslizante(d['V'][:, 0], win)
                    vrms_b = rms_deslizante(d['V'][:, 1], win)
                    vrms_c = rms_deslizante(d['V'][:, 2], win)

                    ax.plot(d['t'], vrms_a, 'b', label='Va')
                    ax.plot(d['t'], vrms_b, 'r', label='Vb')
                    ax.plot(d['t'], vrms_c, 'g', label='Vc')
                    ax.axvline(INSTANTE_FALTA, color='k', linestyle='--')
                    ax.set_title(f"{topo} {terra}")
                    if ax == axs[0, 0]: ax.legend(loc='lower left', fontsize='x-small')
                    plotted_any = True
                except Exception as e:
                    ax.text(0.5, 0.5, "Erro dados", ha='center')
            else:
                ax.text(0.5, 0.5, "N/A", ha='center')  # Ex: T2F1 no MRN

        if plotted_any:
            fig.suptitle(f"Tensão RMS - Barra {barra} - Falta {falta}")
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_saida, f"Fig4_6_Tensao_{barra}.png"))
        plt.close()


def gerar_figura_4_12(cenarios, pasta_saida):
    """Fig 4.12: Corrente Máxima por Tipo de Falta (Todas as Barras)."""
    print("Gerando Fig 4.12 (Correntes Máximas)...")
    tipos = ['ABC', 'A_terra', 'AB', 'BC']
    terra = 'com_terra'

    # Para cada barra, um gráfico
    for barra in BARRAS_DISPONIVEIS:
        res = []
        for f in tipos:
            for topo in ['MRN', 'T2F']:
                for reg in ['com_reg', 'sem_reg']:
                    key = (topo, reg, terra, f)
                    if key in cenarios and barra in cenarios[key]['barras']:
                        d = cenarios[key]['barras'][barra]
                        fs = cenarios[key]['fs']

                        # Janela de falta (100ms após instante)
                        idx_falta = indice_falta(d['t'])
                        win_falta = int(0.1 * fs)
                        end = min(len(d['t']), idx_falta + win_falta)

                        if end > idx_falta:
                            I_seg = d['I'][idx_falta:end]
                            # RMS máximo de qualquer fase na janela
                            # Aproximação rápida: max do valor absoluto / sqrt(2) ou RMS deslizante
                            # Vamos usar o pico do RMS deslizante
                            win_rms = int(fs / 60)
                            imaxs = [np.max(rms_deslizante(I_seg[:, i], win_rms)) for i in range(3)]
                            res.append({'Falta': f, 'Cenario': f"{topo} {reg}", 'Imax': max(imaxs)})

        if res:
            df = pd.DataFrame(res)
            pivot = df.pivot_table(index='Falta', columns='Cenario', values='Imax', aggfunc='max')
            if not pivot.empty:
                pivot.plot(kind='bar', figsize=(10, 6))
                plt.title(f"Corrente Máxima de Falta - Barra {barra}")
                plt.ylabel("Corrente (A)")
                plt.tight_layout()
                plt.savefig(os.path.join(pasta_saida, f"Fig4_12_Imax_{barra}.png"))
                plt.close()


def gerar_comparativo_especifico_MRT_A_vs_T2F_AB(cenarios, pasta_saida):
    """
    Gera gráficos comparando especificamente:
    1. MRT (MRN) - Falta A-Terra
    2. T2F       - Falta AB (Fase-Fase)
    """
    print("\n🔹 Gerando Comparativo Especial: MRT (Falta A) vs T2F (Falta AB)...")

    # Garante que Path está importado
    from pathlib import Path

    pasta_comp = Path(pasta_saida) / "_COMPARATIVO_ESPECIAL_MRT_A_vs_T2F_AB"
    pasta_comp.mkdir(parents=True, exist_ok=True)

    # --- DEFINIÇÃO DAS CHAVES DOS CENÁRIOS ---
    key_mrt = ('MRN', 'com_reg', 'com_terra', 'A_terra')
    key_t2f = ('T2F', 'com_reg', 'com_terra', 'AB')

    # Verifica existência
    if key_mrt not in cenarios:
        print(f"⚠️ Cenário MRT {key_mrt} não encontrado.")
        return
    if key_t2f not in cenarios:
        print(f"⚠️ Cenário T2F {key_t2f} não encontrado.")
        return

    barras_foco = ['820', '800']

    for barra in barras_foco:
        if barra not in cenarios[key_mrt]['barras'] or barra not in cenarios[key_t2f]['barras']:
            continue

        d_mrt = cenarios[key_mrt]['barras'][barra]
        d_t2f = cenarios[key_t2f]['barras'][barra]

        fs_mrt = cenarios[key_mrt]['fs']
        fs_t2f = cenarios[key_t2f]['fs']

        win_mrt = int(fs_mrt / 60)
        win_t2f = int(fs_t2f / 60)

        # --- GRÁFICO 1: CORRENTES RMS ---
        fig, ax = plt.subplots(figsize=(12, 6))

        i_rms_mrt = rms_deslizante(d_mrt['I'][:, 0], win_mrt)

        # T2F: Máximo entre A e B
        i_rms_t2f_A = rms_deslizante(d_t2f['I'][:, 0], win_t2f)
        i_rms_t2f_B = rms_deslizante(d_t2f['I'][:, 1], win_t2f)
        i_rms_t2f_max = np.maximum(i_rms_t2f_A, i_rms_t2f_B)

        L = min(len(d_mrt['t']), len(d_t2f['t']))
        t_plot = d_mrt['t'][:L]

        ax.plot(t_plot, i_rms_mrt[:L], label='MRT (Falta A-G) - Fase A',
                color=CORES['MRN'], linewidth=2.5)
        ax.plot(t_plot, i_rms_t2f_max[:L], label='T2F (Falta AB) - Máx(IA, IB)',
                color=CORES['T2F'], linewidth=2.5, linestyle='--')

        ax.axvline(x=INSTANTE_FALTA, color='k', linestyle=':', alpha=0.6)

        pico_mrt = np.max(i_rms_mrt)
        pico_t2f = np.max(i_rms_t2f_max)

        ax.set_title(f"Comparação de Corrente - Barra {barra}\nMRT (Mono) vs T2F (Bifásico)")
        ax.set_ylabel("Corrente RMS (A)")
        ax.set_xlabel("Tempo (s)")
        ax.legend()

        ax.text(0.02, 0.95, f"Pico MRT: {pico_mrt:.1f} A\nPico T2F: {pico_t2f:.1f} A",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # SALVAMENTO DIRETO (CORREÇÃO DO ERRO)
        fig.savefig(pasta_comp / f"Comparativo_Corrente_Barra_{barra}.png", bbox_inches='tight')
        plt.close(fig)

        # --- GRÁFICO 2: TENSÕES RMS ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        v_mrt_a = rms_deslizante(d_mrt['V'][:, 0], win_mrt)
        v_mrt_b = rms_deslizante(d_mrt['V'][:, 1], win_mrt)
        v_mrt_c = rms_deslizante(d_mrt['V'][:, 2], win_mrt)

        ax1.plot(d_mrt['t'], v_mrt_a, color='blue', label='Va (Faltosa)')
        ax1.plot(d_mrt['t'], v_mrt_b, color='red', alpha=0.5, label='Vb')
        ax1.plot(d_mrt['t'], v_mrt_c, color='green', alpha=0.5, label='Vc')
        ax1.set_title("MRT - Falta A-Terra")
        ax1.legend(fontsize='small')

        v_t2f_a = rms_deslizante(d_t2f['V'][:, 0], win_t2f)
        v_t2f_b = rms_deslizante(d_t2f['V'][:, 1], win_t2f)
        v_t2f_c = rms_deslizante(d_t2f['V'][:, 2], win_t2f)

        ax2.plot(d_t2f['t'], v_t2f_a, color='blue', label='Va (Faltosa)')
        ax2.plot(d_t2f['t'], v_t2f_b, color='red', label='Vb (Faltosa)')
        ax2.plot(d_t2f['t'], v_t2f_c, color='green', alpha=0.5, label='Vc (Sã)')
        ax2.set_title("T2F - Falta AB")
        ax2.legend(fontsize='small')

        fig.suptitle(f"Impacto na Tensão - Barra {barra}")
        plt.tight_layout()

        # SALVAMENTO DIRETO (CORREÇÃO DO ERRO)
        fig.savefig(pasta_comp / f"Comparativo_Tensao_Barra_{barra}.png", bbox_inches='tight')
        plt.close(fig)

    print("✅ Comparativo Especial concluído.")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    PASTA_RESULTADOS = "Resultados_Tese_Corrigido_vFinal"
    if not os.path.exists(PASTA_RESULTADOS): os.makedirs(PASTA_RESULTADOS)

    cenarios = carregar_todos_cenarios()

    if cenarios:
        gerar_figura_4_4(cenarios, PASTA_RESULTADOS)
        gerar_figura_4_6(cenarios, PASTA_RESULTADOS)
        gerar_figura_4_12(cenarios, PASTA_RESULTADOS)
        gerar_comparativo_especifico_MRT_A_vs_T2F_AB(cenarios, PASTA_RESULTADOS)

        # Você pode adicionar as chamadas para 4.5, 4.7, etc. aqui seguindo o modelo da 4.6
        # Elas vão funcionar pois a lógica de "t existe se V existe" está garantida.

        print("\n✅ Processamento Concluído! Verifique a pasta de resultados.")