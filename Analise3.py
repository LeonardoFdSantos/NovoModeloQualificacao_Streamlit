"""
================================================================================
ANÁLISE CAPÍTULO 4 - TESTES DE CURTO-CIRCUITO
Sistema IEEE 34 Barras - MRT (SWER) vs T2F
Versão Profissional para Tese de Doutorado
================================================================================
Autor: Leonardo Santos
Data: Janeiro 2026
================================================================================
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuração visual profissional
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

# Cores personalizadas para a tese
CORES = {
    'MRT': '#E74C3C',      # Vermelho
    'T2F': '#3498DB',      # Azul
    'Fase_A': '#00CED1',   # Cyan
    'Fase_B': '#FF6347',   # Tomate
    'Fase_C': '#32CD32',   # Verde Lima
    'I0': '#4169E1',       # Azul Royal
    'I1': '#228B22',       # Verde Floresta
    'I2': '#8B008B'        # Magenta Escuro
}

# ==============================================================================
# 1. PROCESSADOR DE SINAIS
# ==============================================================================
class ProcessadorSinais:
    """Processa sinais brutos do MATLAB/Simulink (corrente/tensão por fase)"""
    
    def __init__(self, arquivo_mat, freq=60):
        self.arquivo = arquivo_mat
        self.freq = freq
        self.dados = {}
        self._carregar_dados()
        self._processar_sinais()
    
    def _carregar_dados(self):
        try:
            with h5py.File(self.arquivo, 'r') as f:
                # tempo
                if 't' in f.keys():
                    self.t = np.array(f['t']).flatten()
                else:
                    raise ValueError("Arquivo sem vetor 't'")
                
                # posição da falta (se existir)
                self.m1 = float(f['m1'][()]) if 'm1' in f.keys() else 0.5
                
                # todos os sinais *_raw
                for key in f.keys():
                    if key.endswith('_raw'):
                        nome = key.replace('_raw', '')
                        dados_raw = np.array(f[key])
                        
                        # Ajustar dimensões (N x 3)
                        if dados_raw.ndim == 1:
                            dados_raw = np.column_stack([dados_raw]*3)
                        elif dados_raw.shape[0] == 3 and dados_raw.shape[1] > 3:
                            dados_raw = dados_raw.T
                        
                        L = min(len(self.t), len(dados_raw))
                        self.t = self.t[:L]
                        dados_raw = dados_raw[:L]
                        
                        self.dados[nome] = {
                            'raw': dados_raw,
                            'rms': None,
                            'seq': None
                        }
            
            print(f"✓ {Path(self.arquivo).stem[:90]}")
            
        except Exception as e:
            raise Exception(f"Erro ao carregar: {e}")
    
    def _processar_sinais(self):
        """Calcula RMS deslizante e componentes simétricas (quando for corrente)"""
        dt = self.t[1] - self.t[0]
        self.fs = 1.0 / dt
        self.samples_ciclo = max(1, int(self.fs / self.freq))
        
        for nome, sinal in self.dados.items():
            raw = sinal['raw']
            
            # RMS deslizante
            try:
                sinal['rms'] = np.sqrt(np.abs(
                    scipy.ndimage.uniform_filter1d(raw**2, self.samples_ciclo, axis=0)
                ))
            except Exception:
                sinal['rms'] = np.zeros_like(raw)
            
            # Componentes simétricas apenas para correntes (nome começando com I_)
            if nome.startswith('I_'):
                try:
                    sinal['seq'] = self._calcular_seq(raw)
                except Exception:
                    sinal['seq'] = None
    
    def _calcular_seq(self, abc):
        """Componentes simétricas via transformada de Fortescue"""
        a = np.exp(1j * 2*np.pi / 3)
        rot = np.exp(-1j * 2*np.pi * self.freq * self.t)
        
        fasores = np.zeros((len(self.t), 3), dtype=complex)
        for i in range(3):
            fasores[:, i] = scipy.ndimage.uniform_filter1d(
                abc[:, i] * rot, self.samples_ciclo
            ) * np.sqrt(2)
        
        I0 = (fasores[:,0] + fasores[:,1] + fasores[:,2]) / 3.0
        I1 = (fasores[:,0] + a*fasores[:,1] + a**2*fasores[:,2]) / 3.0
        I2 = (fasores[:,0] + a**2*fasores[:,1] + a*fasores[:,2]) / 3.0
        
        return {'I0': np.abs(I0), 'I1': np.abs(I1), 'I2': np.abs(I2)}
    
    def detectar_falta(self, sinal_preferencial='I_822'):
        """Detecta instante aproximado de início de falta via aumento de corrente RMS"""
        if sinal_preferencial not in self.dados or self.dados[sinal_preferencial]['rms'] is None:
            # tenta qualquer corrente disponível
            for nome, info in self.dados.items():
                if nome.startswith('I_') and info['rms'] is not None:
                    sinal_preferencial = nome
                    break
            else:
                return len(self.t) // 2
        
        rms = self.dados[sinal_preferencial]['rms']
        i_max = np.max(rms, axis=1)
        baseline = np.median(i_max[:len(i_max)//5])
        idx = np.where(i_max > baseline * 2.0)[0]
        
        return idx[0] if len(idx) > 0 else len(self.t) // 2


# ==============================================================================
# 2. ANALISADOR PRINCIPAL
# ==============================================================================
class AnalisadorCapitulo4:
    """Análise completa do Capítulo 4 da Tese (MRT/SWER vs T2F)"""
    
    def __init__(self, pasta_entrada, pasta_saida='Resultados_Cap4_Final'):
        self.pasta = Path(pasta_entrada)
        self.pasta_saida = Path(pasta_saida)
        self.arquivos = sorted(self.pasta.glob('*_py.mat'))
        self.resultados = []
        self.df = None
        
        # Estrutura de pastas de saída (por sistema/condição)
        pastas = [
            self.pasta_saida / 'CSVs',
            self.pasta_saida / 'Tabelas_LaTeX',
            self.pasta_saida / 'Graficos' / 'MRT_COM_AT',
            self.pasta_saida / 'Graficos' / 'MRT_SEM_AT',
            self.pasta_saida / 'Graficos' / 'T2F_COM_AT',
            self.pasta_saida / 'Graficos' / 'T2F_SEM_AT',
            self.pasta_saida / 'Graficos' / 'Comparacoes',
            self.pasta_saida / 'PDFs'
        ]
        
        for p in pastas:
            p.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*100}")
        print(f"{'ANALISADOR CAPÍTULO 4 - SISTEMA IEEE 34 BARRAS':^100}")
        print(f"{'='*100}")
        print(f"  Entrada:  {self.pasta}")
        print(f"  Saída:    {self.pasta_saida}")
        print(f"  Arquivos: {len(self.arquivos)} casos")
        print(f"{'='*100}\n")
    
    # ----------------------------------------------------------------------
    # Processamento geral
    # ----------------------------------------------------------------------
    def processar_todos(self):
        """Processa todos os arquivos .mat"""
        print(f"{'PROCESSANDO ARQUIVOS':^100}")
        print(f"{'-'*100}\n")
        
        for i, arq in enumerate(self.arquivos, 1):
            print(f"[{i:3d}/{len(self.arquivos)}] ", end='')
            try:
                proc = ProcessadorSinais(str(arq))
                self.resultados.append(self._extrair_metricas(proc, arq.stem))
            except Exception as e:
                print(f"✗ ERRO: {e}")
        
        if len(self.resultados) > 0:
            self.df = pd.DataFrame(self.resultados)
            print(f"\n{'-'*100}")
            print(f"  ✓ PROCESSADO: {len(self.df)} casos com sucesso")
            print(f"{'-'*100}\n")
        else:
            self.df = pd.DataFrame()
            print(f"\n  ✗ Nenhum caso processado\n")
        
        return self.df
    
    # ----------------------------------------------------------------------
    # Extração de metadados a partir do NOME DO ARQUIVO
    # ----------------------------------------------------------------------
    def _extrair_metricas(self, proc, nome_arq):
        """Extrai todas as métricas de um caso a partir dos dados brutos + nome"""
        
        partes = nome_arq.split('__')
        simulacao = partes[0]           # Ex.: "MRT", "MRT_SR", "Qualificacao", "Qualificacao_SR"
        caso = partes[1] if len(partes) > 1 else ''
        
        # ------------------------------------------------------------------
        # Identificação do SISTEMA (MRT/SWER x T2F)
        # ------------------------------------------------------------------
        # MRT = Monofásico com retorno por terra (SWER)
        if simulacao.startswith('MRT'):
            tipo_sistema = 'MRT'    # SWER
        else:
            # tudo que começa com "Qualificacao" é T2F
            tipo_sistema = 'T2F'
        
        # regulador
        com_regulador = 'SR' in simulacao
        
        # aterramento
        sem_aterramento = ('sem_terra' in simulacao) or ('no_Ground' in simulacao)
        
        # ------------------------------------------------------------------
        # Identificação do TIPO DE FALTA
        # ------------------------------------------------------------------
        if ('Sem_Falta' in caso) or ('Normal' in caso):
            tipo_falta = 'Normal'
        elif 'Falta_ABC' in caso:
            tipo_falta = 'ABC'
        elif 'Falta_AB' in caso:
            tipo_falta = 'AB'
        elif 'Falta_AC' in caso:
            tipo_falta = 'AC'
        elif 'Falta_BC' in caso:
            tipo_falta = 'BC'
        elif 'Falta_A' in caso:
            # para MRT e também para os casos especiais de Qualificacao_sem_terra__A_..._-_Falta_A_py
            tipo_falta = 'A-G'
        else:
            tipo_falta = 'Desconhecido'
        
        # ------------------------------------------------------------------
        # Localização da falta (barra)
        # ------------------------------------------------------------------
        local = 'Indefinido'
        for loc in ['822_Meio', '820_Meio', '818_2_Meio',
                    '822', '820', '818_2', '818_1', '816']:
            if loc in caso:
                local = f'Barra {loc.replace(\"_\", \".\")}'
                break
        
        # índice aproximado da falta (usando corrente da barra 822 se houver)
        idx_falta = proc.detectar_falta('I_822')
        
        resultado = {
            'arquivo': nome_arq,
            'tipo_sistema': tipo_sistema,
            'com_regulador': com_regulador,
            'sem_aterramento': sem_aterramento,
            'condicao_aterramento': 'SEM Aterramento' if sem_aterramento else 'COM Aterramento',
            'tipo_falta': tipo_falta,
            'local_falta': local,
            'm1': float(proc.m1),
            't_falta': float(proc.t[idx_falta]),
            'processador': proc
        }
        
        # ------------------------------------------------------------------
        # Extrai métricas nas barras de interesse (corrente/tensão)
        # ------------------------------------------------------------------
        for ponto in ['800', '816', '818', '820', '822']:
            nome_i = f'I_{ponto}'
            if nome_i in proc.dados and proc.dados[nome_i]['rms'] is not None:
                rms = proc.dados[nome_i]['rms']
                
                resultado[f'{ponto}_I_pico_A'] = float(np.max(rms[:, 0]))
                resultado[f'{ponto}_I_pico_B'] = float(np.max(rms[:, 1]))
                resultado[f'{ponto}_I_pico_C'] = float(np.max(rms[:, 2]))
                resultado[f'{ponto}_I_pico_max'] = float(np.max(rms))
                
                if proc.dados[nome_i]['seq']:
                    seq = proc.dados[nome_i]['seq']
                    resultado[f'{ponto}_I0'] = float(seq['I0'][idx_falta])
                    resultado[f'{ponto}_I1'] = float(seq['I1'][idx_falta])
                    resultado[f'{ponto}_I2'] = float(seq['I2'][idx_falta])
            
            nome_v = f'V_{ponto}'
            if nome_v in proc.dados and proc.dados[nome_v]['rms'] is not None:
                v_rms = proc.dados[nome_v]['rms']
                resultado[f'{ponto}_V_min'] = float(np.min(v_rms[idx_falta, :]))
                resultado[f'{ponto}_V_max'] = float(np.max(v_rms[idx_falta, :]))
        
        return resultado
    
    # ----------------------------------------------------------------------
    # Tabelas comparativas (CSV + LaTeX)
    # ----------------------------------------------------------------------
    def gerar_tabelas_comparativas(self):
        """Gera tabelas comparativas formatadas para a tese"""
        
        if self.df is None or len(self.df) == 0:
            return
        
        print(f"\n{'GERANDO TABELAS COMPARATIVAS':^100}")
        print(f"{'-'*100}\n")
        
        # ===============================================================
        # TABELA 0: Lista de casos de simulação (para início do capítulo)
        # ===============================================================
        cols_meta = ['arquivo', 'tipo_sistema', 'condicao_aterramento',
                     'tipo_falta', 'local_falta', 'm1']
        df_casos = self.df[cols_meta].sort_values(['tipo_sistema', 'condicao_aterramento',
                                                   'tipo_falta', 'local_falta', 'm1'])
        df_casos.to_csv(self.pasta_saida / 'CSVs' / 'Tabela_0_Casos_Simulacao.csv',
                        index=False, encoding='utf-8-sig')
        
        # ===============================================================
        # TABELA 1: Resumo geral por sistema e tipo de falta (barra 822)
        # ===============================================================
        tabela1_data = []
        
        for sistema in ['MRT', 'T2F']:
            df_sistema = self.df[self.df['tipo_sistema'] == sistema]
            tipos_falta = df_sistema['tipo_falta'].unique()
            
            for tipo_falta in tipos_falta:
                if tipo_falta in ['Desconhecido', 'Normal']:
                    continue
                
                df_tipo = df_sistema[df_sistema['tipo_falta'] == tipo_falta]
                
                if len(df_tipo) > 0 and '822_I_pico_max' in df_tipo.columns:
                    tabela1_data.append({
                        'Sistema': sistema,
                        'Tipo de Falta': tipo_falta,
                        'N° Casos': len(df_tipo),
                        'I_pico_min (A)': df_tipo['822_I_pico_max'].min(),
                        'I_pico_max (A)': df_tipo['822_I_pico_max'].max(),
                        'I_pico_média (A)': df_tipo['822_I_pico_max'].mean(),
                        'I_pico_std (A)': df_tipo['822_I_pico_max'].std(),
                        'I0_média (A)': df_tipo['822_I0'].mean() if '822_I0' in df_tipo.columns else 0,
                        'I1_média (A)': df_tipo['822_I1'].mean() if '822_I1' in df_tipo.columns else 0,
                        'I2_média (A)': df_tipo['822_I2'].mean() if '822_I2' in df_tipo.columns else 0
                    })
        
        df_tab1 = pd.DataFrame(tabela1_data)
        
        df_tab1.to_csv(self.pasta_saida / 'CSVs' / 'Tabela_1_Resumo_Geral.csv',
                       index=False, encoding='utf-8-sig', float_format='%.2f')
        
        with open(self.pasta_saida / 'Tabelas_LaTeX' / 'Tabela_1_Resumo_Geral.tex',
                  'w', encoding='utf-8') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Resumo estatístico das correntes de curto-circuito na barra 822}\n")
            f.write("\\label{tab:resumo_geral_cap4}\n")
            f.write(df_tab1.to_latex(index=False, float_format='%.2f'))
            f.write("\\end{table}\n")
        
        print("  ✓ Tabela 1: Resumo Geral (barra 822)")
        
        # ===============================================================
        # TABELA 2: Comparação direta MRT (A-G) vs T2F (ABC, AB, AC, BC)
        #          em m1 ≈ 0.5 e barra 822
        # ===============================================================
        print(f"\n{'TABELA 2 - COMPARAÇÃO T2F vs MRT':^100}")
        print(f"{'-'*100}\n")
        
        tabela2_data = []
        
        for com_terra in [True, False]:
            terra_label = 'COM Aterramento' if com_terra else 'SEM Aterramento'
            
            df_mrt = self.df[
                (self.df['tipo_sistema'] == 'MRT') &
                (self.df['tipo_falta'] == 'A-G') &
                (self.df['sem_aterramento'] == (not com_terra)) &
                (np.abs(self.df['m1'] - 0.5) < 0.05)
            ]
            
            for tipo_t2f in ['ABC', 'AB', 'AC', 'BC']:
                df_t2f = self.df[
                    (self.df['tipo_sistema'] == 'T2F') &
                    (self.df['tipo_falta'] == tipo_t2f) &
                    (self.df['sem_aterramento'] == (not com_terra)) &
                    (np.abs(self.df['m1'] - 0.5) < 0.05)
                ]
                
                if len(df_mrt) > 0 and len(df_t2f) > 0:
                    for col, label in [('822_I_pico_max', 'I_pico'),
                                       ('822_I0', 'I0'),
                                       ('822_I1', 'I1'),
                                       ('822_I2', 'I2')]:
                        if col in df_mrt.columns and col in df_t2f.columns:
                            val_mrt = df_mrt[col].mean()
                            val_t2f = df_t2f[col].mean()
                            delta_abs = val_t2f - val_mrt
                            delta_pct = (delta_abs / val_mrt * 100) if val_mrt != 0 else 0
                            
                            tabela2_data.append({
                                'Condição': terra_label,
                                'Falta_MRT': 'A-G',
                                'Falta_T2F': tipo_t2f,
                                'Métrica': label,
                                'MRT (A)': val_mrt,
                                'T2F (A)': val_t2f,
                                'Δ Absoluto (A)': delta_abs,
                                'Δ Relativo (%)': delta_pct
                            })
        
        df_tab2 = pd.DataFrame(tabela2_data)
        
        df_tab2.to_csv(self.pasta_saida / 'CSVs' / 'Tabela_2_Comparacao_MRT_vs_T2F.csv',
                       index=False, encoding='utf-8-sig', float_format='%.2f')
        
        with open(self.pasta_saida / 'Tabelas_LaTeX' / 'Tabela_2_Comparacao_MRT_vs_T2F.tex',
                  'w', encoding='utf-8') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Comparação quantitativa entre MRT (A-G) e T2F (ABC, AB, AC, BC) na barra 822, m1 \\approx 0{,}5}\n")
            f.write("\\label{tab:comparacao_t2f_mrt_cap4}\n")
            f.write(df_tab2.to_latex(index=False, float_format='%.2f'))
            f.write("\\end{table}\n")
        
        print("  ✓ Tabela 2: Comparação MRT (SWER) vs T2F")
        
        print(f"\n{'-'*100}")
        print("  ✓ Tabelas salvas em: CSVs/ e Tabelas_LaTeX/")
        print(f"{'-'*100}\n")
    
    # ----------------------------------------------------------------------
    # Gráficos profissionais (separando por sistema/condição)
    # ----------------------------------------------------------------------
    def gerar_graficos_profissionais(self):
        """Gera gráficos de qualidade para publicação"""
        
        if self.df is None or len(self.df) == 0:
            return
        
        print(f"\n{'GERANDO GRÁFICOS PROFISSIONAIS':^100}")
        print(f"{'-'*100}\n")
        
        # ===============================================================
        # MRT (SWER) - Falta A-G
        # ===============================================================
        print("  Gerando gráficos MRT (SWER)...")
        
        df_mrt = self.df[(self.df['tipo_sistema'] == 'MRT') &
                         (self.df['tipo_falta'] == 'A-G')]
        
        for com_terra in [True, False]:
            df_sub = df_mrt[df_mrt['sem_aterramento'] == (not com_terra)]
            if len(df_sub) < 2:
                continue
            
            terra_label = 'COM_Aterramento' if com_terra else 'SEM_Aterramento'
            pasta_fig = self.pasta_saida / 'Graficos' / ('MRT_COM_AT' if com_terra else 'MRT_SEM_AT')
            
            # Agrupar por local de falta
            for local in sorted(df_sub['local_falta'].unique()):
                df_loc = df_sub[df_sub['local_falta'] == local].sort_values('m1')
                if len(df_loc) < 2:
                    continue
                
                fig = plt.figure(figsize=(18, 12))
                gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
                
                # Correntes por fase
                ax1 = fig.add_subplot(gs[0, :])
                ax1.plot(df_loc['m1']*100, df_loc['822_I_pico_A'],
                         'o-', color=CORES['Fase_A'], label='Fase A (faltosa)',
                         linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                ax1.plot(df_loc['m1']*100, df_loc['822_I_pico_B'],
                         's--', color=CORES['Fase_B'], label='Fase B (sã)',
                         linewidth=2, markersize=8, alpha=0.7)
                ax1.plot(df_loc['m1']*100, df_loc['822_I_pico_C'],
                         '^--', color=CORES['Fase_C'], label='Fase C (sã)',
                         linewidth=2, markersize=8, alpha=0.7)
                
                ax1.set_xlabel('Posição da falta (% do comprimento da linha)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Corrente de pico RMS (A)', fontsize=14, fontweight='bold')
                ax1.set_title(f'MRT (SWER) - Falta A-G ({terra_label}) - {local} - Correntes trifásicas na barra 822',
                              fontsize=16, fontweight='bold', pad=15)
                ax1.legend(loc='best', framealpha=0.9, shadow=True)
                ax1.grid(True, alpha=0.3, linestyle='--')
                
                # Componentes simétricas
                ax2 = fig.add_subplot(gs[1, :])
                if '822_I0' in df_loc.columns:
                    ax2.plot(df_loc['m1']*100, df_loc['822_I0'],
                             'o-', color=CORES['I0'], label='I₀ (seq. zero)',
                             linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                    ax2.plot(df_loc['m1']*100, df_loc['822_I1'],
                             's-', color=CORES['I1'], label='I₁ (seq. positiva)',
                             linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                    ax2.plot(df_loc['m1']*100, df_loc['822_I2'],
                             '^-', color=CORES['I2'], label='I₂ (seq. negativa)',
                             linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                
                ax2.set_xlabel('Posição da falta (% do comprimento da linha)', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Corrente de sequência (A)', fontsize=14, fontweight='bold')
                ax2.set_title('Componentes simétricas (Fortescue)',
                              fontsize=16, fontweight='bold', pad=15)
                ax2.legend(loc='best', framealpha=0.9, shadow=True)
                ax2.grid(True, alpha=0.3, linestyle='--')
                
                # Tensões
                ax3 = fig.add_subplot(gs[2, 0])
                if '822_V_min' in df_loc.columns:
                    ax3.plot(df_loc['m1']*100, df_loc['822_V_min'],
                             'o-', color='#E74C3C', label='V_min',
                             linewidth=2.5, markersize=9)
                    ax3.plot(df_loc['m1']*100, df_loc['822_V_max'],
                             's-', color='#27AE60', label='V_max',
                             linewidth=2.5, markersize=9)
                
                ax3.set_xlabel('Posição da falta (%)', fontsize=13)
                ax3.set_ylabel('Tensão (V)', fontsize=13)
                ax3.set_title('Afundamento de tensão na barra 822', fontsize=14, fontweight='bold')
                ax3.legend(loc='best', framealpha=0.9)
                ax3.grid(True, alpha=0.3, linestyle='--')
                
                # Razão I0/I1
                ax4 = fig.add_subplot(gs[2, 1])
                if '822_I0' in df_loc.columns and '822_I1' in df_loc.columns:
                    razao = df_loc['822_I0'] / (df_loc['822_I1'] + 1e-6)
                    ax4.plot(df_loc['m1']*100, razao,
                             'o-', color='#9B59B6', linewidth=2.5, markersize=9)
                    ax4.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                                label='I₀ = I₁ (referência)')
                
                ax4.set_xlabel('Posição da falta (%)', fontsize=13)
                ax4.set_ylabel('Razão I₀/I₁', fontsize=13)
                ax4.set_title('Caracterização da falta monofásica A-G (SWER)', fontsize=14, fontweight='bold')
                ax4.legend(loc='best', framealpha=0.9)
                ax4.grid(True, alpha=0.3, linestyle='--')
                
                plt.suptitle(f'Sistema MRT (SWER) - Falta A-G - {terra_label} - {local}',
                             fontsize=18, fontweight='bold', y=0.995)
                
                nome_fig = f'MRT_AG_{terra_label}_{local.replace(\" \", \"_\")}.png'
                plt.savefig(pasta_fig / nome_fig,
                            dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"    ✓ {nome_fig}")
        
        # ===============================================================
        # T2F - Faltas ABC, AB, AC, BC
        # ===============================================================
        print("\n  Gerando gráficos T2F...")
        
        df_t2f = self.df[self.df['tipo_sistema'] == 'T2F']
        
        for tipo_falta in ['ABC', 'AB', 'AC', 'BC']:
            df_falta = df_t2f[df_t2f['tipo_falta'] == tipo_falta]
            if len(df_falta) == 0:
                continue
            
            for com_terra in [True, False]:
                df_sub = df_falta[df_falta['sem_aterramento'] == (not com_terra)]
                if len(df_sub) < 2:
                    continue
                
                terra_label = 'COM_Aterramento' if com_terra else 'SEM_Aterramento'
                pasta_fig = self.pasta_saida / 'Graficos' / ('T2F_COM_AT' if com_terra else 'T2F_SEM_AT')
                
                for local in sorted(df_sub['local_falta'].unique()):
                    df_loc = df_sub[df_sub['local_falta'] == local].sort_values('m1')
                    if len(df_loc) < 2:
                        continue
                    
                    fig = plt.figure(figsize=(18, 12))
                    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
                    
                    # Correntes por fase
                    ax1 = fig.add_subplot(gs[0, :])
                    ax1.plot(df_loc['m1']*100, df_loc['822_I_pico_A'],
                             'o-', color=CORES['Fase_A'], label='Fase A',
                             linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                    ax1.plot(df_loc['m1']*100, df_loc['822_I_pico_B'],
                             's-', color=CORES['Fase_B'], label='Fase B',
                             linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                    ax1.plot(df_loc['m1']*100, df_loc['822_I_pico_C'],
                             '^-', color=CORES['Fase_C'], label='Fase C',
                             linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                    
                    ax1.set_xlabel('Posição da falta (% do comprimento da linha)', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Corrente de pico RMS (A)', fontsize=14, fontweight='bold')
                    ax1.set_title(f'T2F - Falta {tipo_falta} ({terra_label}) - {local} - Correntes trifásicas na barra 822',
                                  fontsize=16, fontweight='bold', pad=15)
                    ax1.legend(loc='best', framealpha=0.9, shadow=True)
                    ax1.grid(True, alpha=0.3, linestyle='--')
                    
                    # Componentes simétricas
                    ax2 = fig.add_subplot(gs[1, :])
                    if '822_I0' in df_loc.columns:
                        ax2.plot(df_loc['m1']*100, df_loc['822_I0'],
                                 'o-', color=CORES['I0'], label='I₀ (seq. zero)',
                                 linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                        ax2.plot(df_loc['m1']*100, df_loc['822_I1'],
                                 's-', color=CORES['I1'], label='I₁ (seq. positiva)',
                                 linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                        ax2.plot(df_loc['m1']*100, df_loc['822_I2'],
                                 '^-', color=CORES['I2'], label='I₂ (seq. negativa)',
                                 linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                    
                    ax2.set_xlabel('Posição da falta (%)', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('Corrente de sequência (A)', fontsize=14, fontweight='bold')
                    ax2.set_title('Componentes simétricas', fontsize=16, fontweight='bold', pad=15)
                    ax2.legend(loc='best', framealpha=0.9, shadow=True)
                    ax2.grid(True, alpha=0.3, linestyle='--')
                    
                    # Desequilíbrio
                    ax3 = fig.add_subplot(gs[2, 0])
                    if '822_I_pico_A' in df_loc.columns:
                        i_avg = (df_loc['822_I_pico_A'] +
                                 df_loc['822_I_pico_B'] +
                                 df_loc['822_I_pico_C']) / 3
                        deseq = np.abs(df_loc['822_I_pico_max'] - i_avg) / (i_avg + 1e-6) * 100
                        ax3.plot(df_loc['m1']*100, deseq, 'o-', color='#E67E22',
                                 linewidth=2.5, markersize=9)
                    
                    ax3.set_xlabel('Posição da falta (%)', fontsize=13)
                    ax3.set_ylabel('Desequilíbrio (%)', fontsize=13)
                    ax3.set_title('Desequilíbrio entre fases', fontsize=14, fontweight='bold')
                    ax3.grid(True, alpha=0.3, linestyle='--')
                    
                    # Corrente máxima
                    ax4 = fig.add_subplot(gs[2, 1])
                    ax4.plot(df_loc['m1']*100, df_loc['822_I_pico_max'],
                             'o-', color=CORES['T2F'], linewidth=3, markersize=10)
                    
                    ax4.set_xlabel('Posição da falta (%)', fontsize=13)
                    ax4.set_ylabel('I_pico_max (A)', fontsize=13)
                    ax4.set_title('Corrente máxima de falta', fontsize=14, fontweight='bold')
                    ax4.grid(True, alpha=0.3, linestyle='--')
                    
                    plt.suptitle(f'Sistema T2F - Falta {tipo_falta} - {terra_label} - {local}',
                                 fontsize=18, fontweight='bold', y=0.995)
                    
                    nome_fig = f'T2F_{tipo_falta}_{terra_label}_{local.replace(\" \", \"_\")}.png'
                    plt.savefig(pasta_fig / nome_fig,
                                dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
                    print(f"    ✓ {nome_fig}")
        
        # ===============================================================
        # Gráficos comparativos MRT vs T2F (barra 822)
        # ===============================================================
        print("\n  Gerando gráficos comparativos MRT vs T2F...")
        
        for com_terra in [True, False]:
            terra_label = 'COM_Aterramento' if com_terra else 'SEM_Aterramento'
            
            df_mrt = self.df[(self.df['tipo_sistema'] == 'MRT') &
                             (self.df['tipo_falta'] == 'A-G') &
                             (self.df['sem_aterramento'] == (not com_terra))]
            df_t2f = self.df[(self.df['tipo_sistema'] == 'T2F') &
                             (self.df['tipo_falta'] == 'ABC') &
                             (self.df['sem_aterramento'] == (not com_terra))]
            
            if len(df_mrt) < 2 or len(df_t2f) < 2:
                continue
            
            df_mrt_sorted = df_mrt.sort_values('m1')
            df_t2f_sorted = df_t2f.sort_values('m1')
            
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            
            # Corrente máxima
            ax = axes[0, 0]
            ax.plot(df_mrt_sorted['m1']*100, df_mrt_sorted['822_I_pico_max'],
                    'o-', color=CORES['MRT'], label='MRT (A-G)', linewidth=3, markersize=10)
            ax.plot(df_t2f_sorted['m1']*100, df_t2f_sorted['822_I_pico_max'],
                    's-', color=CORES['T2F'], label='T2F (ABC)', linewidth=3, markersize=10)
            ax.set_xlabel('Posição da falta (%)', fontsize=13)
            ax.set_ylabel('I_pico_max (A)', fontsize=13)
            ax.set_title('Corrente máxima na barra 822', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # I0
            ax = axes[0, 1]
            if '822_I0' in df_mrt_sorted.columns and '822_I0' in df_t2f_sorted.columns:
                ax.plot(df_mrt_sorted['m1']*100, df_mrt_sorted['822_I0'],
                        'o-', color=CORES['MRT'], label='MRT (A-G)', linewidth=3, markersize=10)
                ax.plot(df_t2f_sorted['m1']*100, df_t2f_sorted['822_I0'],
                        's-', color=CORES['T2F'], label='T2F (ABC)', linewidth=3, markersize=10)
            ax.set_xlabel('Posição da falta (%)', fontsize=13)
            ax.set_ylabel('I₀ (A)', fontsize=13)
            ax.set_title('Componente de sequência zero', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # I1
            ax = axes[1, 0]
            if '822_I1' in df_mrt_sorted.columns and '822_I1' in df_t2f_sorted.columns:
                ax.plot(df_mrt_sorted['m1']*100, df_mrt_sorted['822_I1'],
                        'o-', color=CORES['MRT'], label='MRT (A-G)', linewidth=3, markersize=10)
                ax.plot(df_t2f_sorted['m1']*100, df_t2f_sorted['822_I1'],
                        's-', color=CORES['T2F'], label='T2F (ABC)', linewidth=3, markersize=10)
            ax.set_xlabel('Posição da falta (%)', fontsize=13)
            ax.set_ylabel('I₁ (A)', fontsize=13)
            ax.set_title('Componente de sequência positiva', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # I2
            ax = axes[1, 1]
            if '822_I2' in df_mrt_sorted.columns and '822_I2' in df_t2f_sorted.columns:
                ax.plot(df_mrt_sorted['m1']*100, df_mrt_sorted['822_I2'],
                        'o-', color=CORES['MRT'], label='MRT (A-G)', linewidth=3, markersize=10)
                ax.plot(df_t2f_sorted['m1']*100, df_t2f_sorted['822_I2'],
                        's-', color=CORES['T2F'], label='T2F (ABC)', linewidth=3, markersize=10)
            ax.set_xlabel('Posição da falta (%)', fontsize=13)
            ax.set_ylabel('I₂ (A)', fontsize=13)
            ax.set_title('Componente de sequência negativa', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.suptitle(f'Comparação MRT (SWER) vs T2F - {terra_label} - Barra 822',
                         fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            nome_fig = f'Comparacao_MRT_vs_T2F_{terra_label}.png'
            plt.savefig(self.pasta_saida / 'Graficos' / 'Comparacoes' / nome_fig,
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"    ✓ {nome_fig}")
        
        print(f"\n{'-'*100}")
        print("  ✓ Gráficos salvos em: Graficos/MRT_*, Graficos/T2F_*, Graficos/Comparacoes/")
        print(f"{'-'*100}\n")
    
    # ----------------------------------------------------------------------
    # PDF consolidado
    # ----------------------------------------------------------------------
    def gerar_pdf_consolidado(self):
        """Gera PDF com todos os gráficos"""
        
        print(f"\n{'GERANDO PDF CONSOLIDADO':^100}")
        print(f"{'-'*100}\n")
        
        imgs = []
        for sub in ['MRT_COM_AT', 'MRT_SEM_AT', 'T2F_COM_AT', 'T2F_SEM_AT', 'Comparacoes']:
            imgs += sorted((self.pasta_saida / 'Graficos' / sub).glob('*.png'))
        
        if len(imgs) == 0:
            print("  ✗ Nenhum gráfico encontrado")
            return
        
        pdf_path = self.pasta_saida / 'PDFs' / 'Relatorio_Completo_Cap4.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Página título
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor('white')
            
            fig.text(0.5, 0.70, 'RELATÓRIO CAPÍTULO 4',
                     ha='center', fontsize=28, fontweight='bold')
            fig.text(0.5, 0.62, 'Análise de curtocircuitos - Sistema IEEE 34 barras',
                     ha='center', fontsize=18)
            fig.text(0.5, 0.54, 'Comparação MRT (SWER) vs T2F',
                     ha='center', fontsize=16, style='italic')
            
            fig.text(0.5, 0.40, 'MRT (SWER): faltas monofásicas A-G',
                     ha='center', fontsize=13, color=CORES['MRT'])
            fig.text(0.5, 0.35, 'T2F: faltas ABC, AB, AC, BC',
                     ha='center', fontsize=13, color=CORES['T2F'])
            
            fig.text(0.5, 0.24, f'Total de figuras: {len(imgs)}',
                     ha='center', fontsize=11)
            fig.text(0.5, 0.20, f'Gerado em: {pd.Timestamp.now().strftime("%d/%m/%Y %H:%M")}',
                     ha='center', fontsize=10, style='italic')
            
            plt.axis('off')
            pdf.savefig(fig, dpi=150)
            plt.close()
            
            # Páginas com gráficos
            for img in imgs:
                fig = plt.figure(figsize=(11, 8.5))
                fig.patch.set_facecolor('white')
                img_data = plt.imread(img)
                plt.imshow(img_data)
                plt.axis('off')
                plt.tight_layout(pad=0)
                pdf.savefig(fig, dpi=150)
                plt.close()
        
        print(f"  ✓ PDF gerado: {pdf_path.name} ({len(imgs)+1} páginas)")
        print(f"{'-'*100}\n")
    
    # ----------------------------------------------------------------------
    # Relatório completo
    # ----------------------------------------------------------------------
    def gerar_relatorio_completo(self):
        """Gera relatório completo do Capítulo 4"""
        
        print(f"\n{'='*100}")
        print(f"{'GERANDO RELATÓRIO COMPLETO DO CAPÍTULO 4':^100}")
        print(f"{'='*100}\n")
        
        # CSV geral (sem o objeto processador)
        colunas = [c for c in self.df.columns if c != 'processador']
        self.df[colunas].to_csv(
            self.pasta_saida / 'CSVs' / '00_RESUMO_GERAL_TODOS_CASOS.csv',
            index=False, encoding='utf-8-sig'
        )
        print("  ✓ 00_RESUMO_GERAL_TODOS_CASOS.csv")
        
        self.gerar_tabelas_comparativas()
        self.gerar_graficos_profissionais()
        self.gerar_pdf_consolidado()
        
        print(f"\n{'='*100}")
        print(f"{'✓ RELATÓRIO COMPLETO GERADO COM SUCESSO':^100}")
        print(f"{'='*100}\n")
        
        # Sumário final
        print(f"\n{'SUMÁRIO DE ARQUIVOS GERADOS':^100}")
        print(f"{'-'*100}")
        
        n_csvs = len(list((self.pasta_saida / 'CSVs').glob('*.csv')))
        n_latex = len(list((self.pasta_saida / 'Tabelas_LaTeX').glob('*.tex')))
        n_mrt = len(list((self.pasta_saida / 'Graficos' / 'MRT_COM_AT').glob('*.png'))) \
                + len(list((self.pasta_saida / 'Graficos' / 'MRT_SEM_AT').glob('*.png')))
        n_t2f = len(list((self.pasta_saida / 'Graficos' / 'T2F_COM_AT').glob('*.png'))) \
                + len(list((self.pasta_saida / 'Graficos' / 'T2F_SEM_AT').glob('*.png')))
        n_comp = len(list((self.pasta_saida / 'Graficos' / 'Comparacoes').glob('*.png')))
        n_pdf = len(list((self.pasta_saida / 'PDFs').glob('*.pdf')))
        
        print(f"  CSVs:              {n_csvs:3d} arquivos")
        print(f"  Tabelas LaTeX:     {n_latex:3d} arquivos")
        print(f"  Gráficos MRT:      {n_mrt:3d} imagens")
        print(f"  Gráficos T2F:      {n_t2f:3d} imagens")
        print(f"  Comparações:       {n_comp:3d} imagens")
        print(f"  PDFs:              {n_pdf:3d} arquivo(s)")
        print(f"{'-'*100}")
        print(f"  Pasta de saída: {self.pasta_saida}")
        print(f"{'='*100}\n")


# ==============================================================================
# 3. EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    
    # Ajuste aqui o caminho da pasta com os .mat brutos
    pasta_entrada = PPath("C:/Users/Leonardo Felipe/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_10/Processados_HDF5")
    pasta_saida = "Resultados_Cap4_Tese_Final"
    
    analisador = AnalisadorCapitulo4(pasta_entrada, pasta_saida)
    df = analisador.processar_todos()
    
    if len(df) > 0:
        analisador.gerar_relatorio_completo()
    else:
        print("\n✗ Nenhum dado processado. Verifique os arquivos de entrada.\n")
