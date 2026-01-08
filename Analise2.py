"""
================================================================================
ANÁLISE CAPÍTULO 4 - TESTES DE CURTO-CIRCUITO
Sistema IEEE 34 Barras - MRT vs T2F
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
    """Processa sinais do MATLAB/Simulink"""
    
    def __init__(self, arquivo_mat, freq=60):
        self.arquivo = arquivo_mat
        self.freq = freq
        self.dados = {}
        self._carregar_dados()
        self._processar_sinais()
    
    def _carregar_dados(self):
        try:
            with h5py.File(self.arquivo, 'r') as f:
                if 't' in f.keys():
                    self.t = np.array(f['t']).flatten()
                else:
                    raise ValueError("Arquivo sem vetor 't'")
                
                self.m1 = float(f['m1'][()]) if 'm1' in f.keys() else 0.5
                
                for key in f.keys():
                    if key.endswith('_raw'):
                        nome = key.replace('_raw', '')
                        dados_raw = np.array(f[key])
                        
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
            
            print(f"✓ {Path(self.arquivo).stem[:70]}")
            
        except Exception as e:
            raise Exception(f"Erro ao carregar: {e}")
    
    def _processar_sinais(self):
        dt = self.t[1] - self.t[0]
        self.fs = 1.0 / dt
        self.samples_ciclo = max(1, int(self.fs / self.freq))
        
        for nome, sinal in self.dados.items():
            raw = sinal['raw']
            
            try:
                sinal['rms'] = np.sqrt(np.abs(
                    scipy.ndimage.uniform_filter1d(raw**2, self.samples_ciclo, axis=0)
                ))
            except:
                sinal['rms'] = np.zeros_like(raw)
            
            if 'I_' in nome:
                try:
                    sinal['seq'] = self._calcular_seq(raw)
                except:
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
    
    def detectar_falta(self, sinal='I_822'):
        if sinal not in self.dados or self.dados[sinal]['rms'] is None:
            return len(self.t) // 2
        
        rms = self.dados[sinal]['rms']
        i_max = np.max(rms, axis=1)
        baseline = np.median(i_max[:len(i_max)//5])
        idx = np.where(i_max > baseline * 2.0)[0]
        
        return idx[0] if len(idx) > 0 else len(self.t) // 2

# ==============================================================================
# 2. ANALISADOR PRINCIPAL
# ==============================================================================
class AnalisadorCapitulo4:
    """Análise completa do Capítulo 4 da Tese"""
    
    def __init__(self, pasta_entrada, pasta_saida='Resultados_Cap4_Final'):
        self.pasta = Path(pasta_entrada)
        self.pasta_saida = Path(pasta_saida)
        self.arquivos = sorted(self.pasta.glob('*_py.mat'))
        self.resultados = []
        self.df = None
        
        # Estrutura de pastas
        pastas = [
            self.pasta_saida / 'CSVs',
            self.pasta_saida / 'Tabelas_LaTeX',
            self.pasta_saida / 'Graficos' / 'MRT',
            self.pasta_saida / 'Graficos' / 'T2F',
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
    def _extrair_metricas(self, proc, nome_arq):
        """Extrai todas as métricas de um caso"""
        
        partes = nome_arq.split('__')
        simulacao = partes[0]
        caso = partes[1] if len(partes) > 1 else ''
        
        # Identificação do sistema
        tipo_sistema = 'T2F' if 'Qualificacao' in simulacao else 'MRT'
        com_regulador = 'SR' in simulacao
        sem_aterramento = 'sem_terra' in simulacao or 'no_Ground' in simulacao
        
        # Identificação do tipo de falta
        if 'Sem_Falta' in caso or 'Normal' in caso:
            tipo_falta = 'Normal'
        elif tipo_sistema == 'MRT':
            tipo_falta = 'A-G' if 'Falta_A' in caso else 'Normal'
        elif tipo_sistema == 'T2F':
            if '_ABC' in caso or 'ABC_' in caso:
                tipo_falta = 'ABC'
            elif '_AB' in caso or 'AB_' in caso:
                tipo_falta = 'AB'
            elif '_AC' in caso or 'AC_' in caso:
                tipo_falta = 'AC'
            elif '_BC' in caso or 'BC_' in caso:
                tipo_falta = 'BC'
            else:
                tipo_falta = 'Normal'
        else:
            tipo_falta = 'Desconhecido'
        
        # Localização
        local = 'Indefinido'
        for loc in ['822', '820', '818_2', '818_1', '816']:
            if loc in caso:
                local = f'Barra {loc.replace("_", ".")}'
                break
        
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
        
        # Extrai métricas de todos os pontos
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
    def gerar_tabelas_comparativas(self):
        """Gera tabelas comparativas formatadas para a tese"""
        
        if self.df is None or len(self.df) == 0:
            return
        
        print(f"\n{'GERANDO TABELAS COMPARATIVAS':^100}")
        print(f"{'-'*100}\n")
        
        # =====================================================================
        # TABELA 1: Resumo Geral por Sistema e Tipo de Falta
        # =====================================================================
        tabela1_data = []
        
        for sistema in ['MRT', 'T2F']:
            df_sistema = self.df[self.df['tipo_sistema'] == sistema]
            tipos_falta = df_sistema['tipo_falta'].unique()
            
            for tipo_falta in tipos_falta:
                if tipo_falta == 'Desconhecido' or tipo_falta == 'Normal':
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
        
        # Salva CSV
        df_tab1.to_csv(self.pasta_saida / 'CSVs' / 'Tabela_1_Resumo_Geral.csv', 
                       index=False, encoding='utf-8-sig', float_format='%.2f')
        
        # Salva LaTeX
        with open(self.pasta_saida / 'Tabelas_LaTeX' / 'Tabela_1_Resumo_Geral.tex', 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Resumo Estatístico das Correntes de Curto-Circuito por Sistema e Tipo de Falta}\n")
            f.write("\\label{tab:resumo_geral_cap4}\n")
            f.write(df_tab1.to_latex(index=False, float_format='%.2f'))
            f.write("\\end{table}\n")
        
        print("  ✓ Tabela 1: Resumo Geral")
        print(df_tab1.to_string(index=False))
        print()
        
        # =====================================================================
        # TABELA 2: Comparação Direta T2F vs MRT
        # =====================================================================
        print(f"\n{'TABELA 2 - COMPARAÇÃO T2F vs MRT':^100}")
        print(f"{'-'*100}\n")
        
        tabela2_data = []
        
        for com_terra in [True, False]:
            terra_label = 'COM Aterramento' if com_terra else 'SEM Aterramento'
            
            # MRT A-G
            df_mrt = self.df[(self.df['tipo_sistema'] == 'MRT') & 
                             (self.df['tipo_falta'] == 'A-G') &
                             (self.df['sem_aterramento'] == (not com_terra)) &
                             (np.abs(self.df['m1'] - 0.5) < 0.05)]
            
            # T2F tipos
            for tipo_t2f in ['ABC', 'AB', 'AC', 'BC']:
                df_t2f = self.df[(self.df['tipo_sistema'] == 'T2F') & 
                                 (self.df['tipo_falta'] == tipo_t2f) &
                                 (self.df['sem_aterramento'] == (not com_terra)) &
                                 (np.abs(self.df['m1'] - 0.5) < 0.05)]
                
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
        
        # Salva
        df_tab2.to_csv(self.pasta_saida / 'CSVs' / 'Tabela_2_Comparacao_MRT_vs_T2F.csv',
                       index=False, encoding='utf-8-sig', float_format='%.2f')
        
        with open(self.pasta_saida / 'Tabelas_LaTeX' / 'Tabela_2_Comparacao_MRT_vs_T2F.tex', 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Comparação Quantitativa T2F vs MRT (m1=0.5, Barra 822)}\n")
            f.write("\\label{tab:comparacao_t2f_mrt_cap4}\n")
            f.write(df_tab2.to_latex(index=False, float_format='%.2f'))  # <- LINHA CORRIGIDA
            f.write("\\end{table}\n")
        
        print("  ✓ Tabela 2: Comparação T2F vs MRT")
        print(df_tab2.head(20).to_string(index=False))
        print(f"  ... ({len(df_tab2)} linhas no total)")
        
        print(f"\n{'-'*100}")
        print("  ✓ Tabelas salvas em: CSVs/ e Tabelas_LaTeX/")
        print(f"{'-'*100}\n")
                                     

    def gerar_graficos_profissionais(self):
        """Gera gráficos de qualidade para publicação"""
        
        if self.df is None or len(self.df) == 0:
            return
        
        print(f"\n{'GERANDO GRÁFICOS PROFISSIONAIS':^100}")
        print(f"{'-'*100}\n")
        
        # =====================================================================
        # GRÁFICOS MRT (A-G)
        # =====================================================================
        print("  Gerando gráficos MRT...")
        
        df_mrt = self.df[self.df['tipo_sistema'] == 'MRT']
        
        for com_terra in [True, False]:
            df_sub = df_mrt[df_mrt['sem_aterramento'] == (not com_terra)]
            
            if len(df_sub) < 2:
                continue
            
            terra_label = 'COM_Aterramento' if com_terra else 'SEM_Aterramento'
            df_sorted = df_sub.sort_values('m1')
            
            # Figura com 3 subplots
            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
            
            # Subplot 1: Correntes por fase
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(df_sorted['m1']*100, df_sorted['822_I_pico_A'], 
                    'o-', color=CORES['Fase_A'], label='Fase A (faltosa)', 
                    linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
            ax1.plot(df_sorted['m1']*100, df_sorted['822_I_pico_B'], 
                    's--', color=CORES['Fase_B'], label='Fase B (sã)', 
                    linewidth=2, markersize=8, alpha=0.7)
            ax1.plot(df_sorted['m1']*100, df_sorted['822_I_pico_C'], 
                    '^--', color=CORES['Fase_C'], label='Fase C (sã)', 
                    linewidth=2, markersize=8, alpha=0.7)
            
            ax1.set_xlabel('Posição da Falta (% do comprimento da linha)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Corrente de Pico RMS (A)', fontsize=14, fontweight='bold')
            ax1.set_title(f'MRT - Falta A-G ({terra_label}) - Correntes Trifásicas na Barra 822',
                         fontsize=16, fontweight='bold', pad=15)
            ax1.legend(loc='best', framealpha=0.9, shadow=True)
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Subplot 2: Componentes simétricas
            ax2 = fig.add_subplot(gs[1, :])
            if '822_I0' in df_sorted.columns:
                ax2.plot(df_sorted['m1']*100, df_sorted['822_I0'], 
                        'o-', color=CORES['I0'], label='I₀ (Sequência Zero)', 
                        linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                ax2.plot(df_sorted['m1']*100, df_sorted['822_I1'], 
                        's-', color=CORES['I1'], label='I₁ (Sequência Positiva)', 
                        linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                ax2.plot(df_sorted['m1']*100, df_sorted['822_I2'], 
                        '^-', color=CORES['I2'], label='I₂ (Sequência Negativa)', 
                        linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
            
            ax2.set_xlabel('Posição da Falta (% do comprimento da linha)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Corrente de Sequência (A)', fontsize=14, fontweight='bold')
            ax2.set_title('Componentes Simétricas (Transformada de Fortescue)',
                         fontsize=16, fontweight='bold', pad=15)
            ax2.legend(loc='best', framealpha=0.9, shadow=True)
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            # Subplot 3: Tensões
            ax3 = fig.add_subplot(gs[2, 0])
            if '822_V_min' in df_sorted.columns:
                ax3.plot(df_sorted['m1']*100, df_sorted['822_V_min'], 
                        'o-', color='#E74C3C', label='V_min', 
                        linewidth=2.5, markersize=9)
                ax3.plot(df_sorted['m1']*100, df_sorted['822_V_max'], 
                        's-', color='#27AE60', label='V_max', 
                        linewidth=2.5, markersize=9)
            
            ax3.set_xlabel('Posição da Falta (%)', fontsize=13)
            ax3.set_ylabel('Tensão (V)', fontsize=13)
            ax3.set_title('Afundamento de Tensão', fontsize=14, fontweight='bold')
            ax3.legend(loc='best', framealpha=0.9)
            ax3.grid(True, alpha=0.3, linestyle='--')
            
            # Subplot 4: Razão I0/I1
            ax4 = fig.add_subplot(gs[2, 1])
            if '822_I0' in df_sorted.columns and '822_I1' in df_sorted.columns:
                razao = df_sorted['822_I0'] / (df_sorted['822_I1'] + 1e-6)
                ax4.plot(df_sorted['m1']*100, razao, 
                        'o-', color='#9B59B6', linewidth=2.5, markersize=9)
                ax4.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                           label='I₀ = I₁ (referência)')
            
            ax4.set_xlabel('Posição da Falta (%)', fontsize=13)
            ax4.set_ylabel('Razão I₀/I₁', fontsize=13)
            ax4.set_title('Caracterização da Falta Monofásica', fontsize=14, fontweight='bold')
            ax4.legend(loc='best', framealpha=0.9)
            ax4.grid(True, alpha=0.3, linestyle='--')
            
            plt.suptitle(f'Sistema MRT - Análise Completa de Falta A-G ({terra_label})',
                        fontsize=18, fontweight='bold', y=0.995)
            
            nome_fig = f'MRT_A-G_{terra_label}_Completo.png'
            plt.savefig(self.pasta_saida / 'Graficos' / 'MRT' / nome_fig, 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"    ✓ {nome_fig}")
        # =====================================================================
        # GRÁFICOS T2F (ABC, AB, AC, BC)
        # =====================================================================
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
                df_sorted = df_sub.sort_values('m1')
                
                # Figura
                fig = plt.figure(figsize=(18, 12))
                gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
                
                # Subplot 1: Correntes por fase
                ax1 = fig.add_subplot(gs[0, :])
                ax1.plot(df_sorted['m1']*100, df_sorted['822_I_pico_A'], 
                        'o-', color=CORES['Fase_A'], label='Fase A', 
                        linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                ax1.plot(df_sorted['m1']*100, df_sorted['822_I_pico_B'], 
                        's-', color=CORES['Fase_B'], label='Fase B', 
                        linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                ax1.plot(df_sorted['m1']*100, df_sorted['822_I_pico_C'], 
                        '^-', color=CORES['Fase_C'], label='Fase C', 
                        linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                
                ax1.set_xlabel('Posição da Falta (% do comprimento da linha)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Corrente de Pico RMS (A)', fontsize=14, fontweight='bold')
                ax1.set_title(f'T2F - Falta {tipo_falta} ({terra_label}) - Correntes Trifásicas na Barra 822',
                             fontsize=16, fontweight='bold', pad=15)
                ax1.legend(loc='best', framealpha=0.9, shadow=True)
                ax1.grid(True, alpha=0.3, linestyle='--')
                
                # Subplot 2: Componentes simétricas
                ax2 = fig.add_subplot(gs[1, :])
                if '822_I0' in df_sorted.columns:
                    ax2.plot(df_sorted['m1']*100, df_sorted['822_I0'], 
                            'o-', color=CORES['I0'], label='I₀ (Zero)', 
                            linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                    ax2.plot(df_sorted['m1']*100, df_sorted['822_I1'], 
                            's-', color=CORES['I1'], label='I₁ (Positiva)', 
                            linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                    ax2.plot(df_sorted['m1']*100, df_sorted['822_I2'], 
                            '^-', color=CORES['I2'], label='I₂ (Negativa)', 
                            linewidth=3, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                
                ax2.set_xlabel('Posição da Falta (%)', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Corrente de Sequência (A)', fontsize=14, fontweight='bold')
                ax2.set_title('Componentes Simétricas', fontsize=16, fontweight='bold', pad=15)
                ax2.legend(loc='best', framealpha=0.9, shadow=True)
                ax2.grid(True, alpha=0.3, linestyle='--')
                
                # Subplot 3: Desequilíbrio
                ax3 = fig.add_subplot(gs[2, 0])
                if '822_I_pico_A' in df_sorted.columns:
                    i_avg = (df_sorted['822_I_pico_A'] + df_sorted['822_I_pico_B'] + df_sorted['822_I_pico_C']) / 3
                    deseq = np.abs(df_sorted['822_I_pico_max'] - i_avg) / (i_avg + 1e-6) * 100
                    ax3.plot(df_sorted['m1']*100, deseq, 'o-', color='#E67E22', linewidth=2.5, markersize=9)
                
                ax3.set_xlabel('Posição da Falta (%)', fontsize=13)
                ax3.set_ylabel('Desequilíbrio (%)', fontsize=13)
                ax3.set_title('Grau de Desequilíbrio entre Fases', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3, linestyle='--')
                
                # Subplot 4: Corrente máxima
                ax4 = fig.add_subplot(gs[2, 1])
                ax4.plot(df_sorted['m1']*100, df_sorted['822_I_pico_max'], 
                        'o-', color=CORES['T2F'], linewidth=3, markersize=10)
                
                ax4.set_xlabel('Posição da Falta (%)', fontsize=13)
                ax4.set_ylabel('I_pico_max (A)', fontsize=13)
                ax4.set_title('Corrente Máxima de Falta', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3, linestyle='--')
                
                plt.suptitle(f'Sistema T2F - Análise Completa de Falta {tipo_falta} ({terra_label})',
                            fontsize=18, fontweight='bold', y=0.995)
                
                nome_fig = f'T2F_{tipo_falta}_{terra_label}_Completo.png'
                plt.savefig(self.pasta_saida / 'Graficos' / 'T2F' / nome_fig, 
                           dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"    ✓ {nome_fig}")
        # =====================================================================
        # GRÁFICOS COMPARATIVOS MRT vs T2F
        # =====================================================================
        print("\n  Gerando gráficos comparativos...")
        
        for com_terra in [True, False]:
            terra_label = 'COM_Aterramento' if com_terra else 'SEM_Aterramento'
            
            # MRT A-G
            df_mrt = self.df[(self.df['tipo_sistema'] == 'MRT') & 
                             (self.df['tipo_falta'] == 'A-G') &
                             (self.df['sem_aterramento'] == (not com_terra))]
            
            # T2F ABC
            df_t2f = self.df[(self.df['tipo_sistema'] == 'T2F') & 
                             (self.df['tipo_falta'] == 'ABC') &
                             (self.df['sem_aterramento'] == (not com_terra))]
            
            if len(df_mrt) < 2 or len(df_t2f) < 2:
                continue
            
            df_mrt_sorted = df_mrt.sort_values('m1')
            df_t2f_sorted = df_t2f.sort_values('m1')
            
            # Figura comparativa
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            
            # Corrente máxima
            ax = axes[0, 0]
            ax.plot(df_mrt_sorted['m1']*100, df_mrt_sorted['822_I_pico_max'], 
                   'o-', color=CORES['MRT'], label='MRT (A-G)', linewidth=3, markersize=10)
            ax.plot(df_t2f_sorted['m1']*100, df_t2f_sorted['822_I_pico_max'], 
                   's-', color=CORES['T2F'], label='T2F (ABC)', linewidth=3, markersize=10)
            ax.set_xlabel('Posição da Falta (%)', fontsize=13)
            ax.set_ylabel('I_pico_max (A)', fontsize=13)
            ax.set_title('Comparação de Corrente Máxima', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # I0
            ax = axes[0, 1]
            if '822_I0' in df_mrt_sorted.columns and '822_I0' in df_t2f_sorted.columns:
                ax.plot(df_mrt_sorted['m1']*100, df_mrt_sorted['822_I0'], 
                       'o-', color=CORES['MRT'], label='MRT (A-G)', linewidth=3, markersize=10)
                ax.plot(df_t2f_sorted['m1']*100, df_t2f_sorted['822_I0'], 
                       's-', color=CORES['T2F'], label='T2F (ABC)', linewidth=3, markersize=10)
            ax.set_xlabel('Posição da Falta (%)', fontsize=13)
            ax.set_ylabel('I₀ (A)', fontsize=13)
            ax.set_title('Sequência Zero', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # I1
            ax = axes[1, 0]
            if '822_I1' in df_mrt_sorted.columns and '822_I1' in df_t2f_sorted.columns:
                ax.plot(df_mrt_sorted['m1']*100, df_mrt_sorted['822_I1'], 
                       'o-', color=CORES['MRT'], label='MRT (A-G)', linewidth=3, markersize=10)
                ax.plot(df_t2f_sorted['m1']*100, df_t2f_sorted['822_I1'], 
                       's-', color=CORES['T2F'], label='T2F (ABC)', linewidth=3, markersize=10)
            ax.set_xlabel('Posição da Falta (%)', fontsize=13)
            ax.set_ylabel('I₁ (A)', fontsize=13)
            ax.set_title('Sequência Positiva', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # I2
            ax = axes[1, 1]
            if '822_I2' in df_mrt_sorted.columns and '822_I2' in df_t2f_sorted.columns:
                ax.plot(df_mrt_sorted['m1']*100, df_mrt_sorted['822_I2'], 
                       'o-', color=CORES['MRT'], label='MRT (A-G)', linewidth=3, markersize=10)
                ax.plot(df_t2f_sorted['m1']*100, df_t2f_sorted['822_I2'], 
                       's-', color=CORES['T2F'], label='T2F (ABC)', linewidth=3, markersize=10)
            ax.set_xlabel('Posição da Falta (%)', fontsize=13)
            ax.set_ylabel('I₂ (A)', fontsize=13)
            ax.set_title('Sequência Negativa', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.suptitle(f'Comparação MRT vs T2F ({terra_label}) - Barra 822',
                        fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            nome_fig = f'Comparacao_MRT_vs_T2F_{terra_label}.png'
            plt.savefig(self.pasta_saida / 'Graficos' / 'Comparacoes' / nome_fig,
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"    ✓ {nome_fig}")
        
        print(f"\n{'-'*100}")
        print("  ✓ Gráficos salvos em: Graficos/MRT/, Graficos/T2F/, Graficos/Comparacoes/")
        print(f"{'-'*100}\n")
    def gerar_pdf_consolidado(self):
        """Gera PDF com todos os gráficos"""
        
        print(f"\n{'GERANDO PDF CONSOLIDADO':^100}")
        print(f"{'-'*100}\n")
        
        # Coleta todas as imagens
        imgs_mrt = sorted((self.pasta_saida / 'Graficos' / 'MRT').glob('*.png'))
        imgs_t2f = sorted((self.pasta_saida / 'Graficos' / 'T2F').glob('*.png'))
        imgs_comp = sorted((self.pasta_saida / 'Graficos' / 'Comparacoes').glob('*.png'))
        
        imgs_todas = list(imgs_mrt) + list(imgs_t2f) + list(imgs_comp)
        
        if len(imgs_todas) == 0:
            print("  ✗ Nenhum gráfico encontrado")
            return
        
        pdf_path = self.pasta_saida / 'PDFs' / 'Relatorio_Completo_Cap4.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Página título
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor('white')
            
            fig.text(0.5, 0.70, 'RELATÓRIO CAPÍTULO 4', 
                    ha='center', fontsize=28, fontweight='bold')
            fig.text(0.5, 0.62, 'Análise de Curtos-Circuito', 
                    ha='center', fontsize=20)
            fig.text(0.5, 0.54, 'Sistema IEEE 34 Barras', 
                    ha='center', fontsize=18, style='italic')
            
            fig.text(0.5, 0.42, 'MRT: Falta Monofásica A-G', 
                    ha='center', fontsize=14, color=CORES['MRT'])
            fig.text(0.5, 0.37, 'T2F: Faltas ABC, AB, AC, BC', 
                    ha='center', fontsize=14, color=CORES['T2F'])
            
            fig.text(0.5, 0.25, f'Total de figuras: {len(imgs_todas)}', 
                    ha='center', fontsize=12)
            fig.text(0.5, 0.20, f'Gerado em: {pd.Timestamp.now().strftime("%d/%m/%Y %H:%M")}', 
                    ha='center', fontsize=10, style='italic')
            
            plt.axis('off')
            pdf.savefig(fig, dpi=150)
            plt.close()
            
            # Páginas com gráficos
            for img in imgs_todas:
                fig = plt.figure(figsize=(11, 8.5))
                fig.patch.set_facecolor('white')
                
                img_data = plt.imread(img)
                plt.imshow(img_data)
                plt.axis('off')
                plt.tight_layout(pad=0)
                
                pdf.savefig(fig, dpi=150)
                plt.close()
        
        print(f"  ✓ PDF gerado: {pdf_path.name} ({len(imgs_todas)+1} páginas)")
        print(f"{'-'*100}\n")

    def gerar_relatorio_completo(self):
        """Gera relatório completo do Capítulo 4"""
        
        print(f"\n{'='*100}")
        print(f"{'GERANDO RELATÓRIO COMPLETO DO CAPÍTULO 4':^100}")
        print(f"{'='*100}\n")
        
        # 1. CSV Geral
        colunas = [c for c in self.df.columns if c != 'processador']
        self.df[colunas].to_csv(
            self.pasta_saida / 'CSVs' / '00_RESUMO_GERAL_TODOS_CASOS.csv',
            index=False, encoding='utf-8-sig'
        )
        print("  ✓ 00_RESUMO_GERAL_TODOS_CASOS.csv")
        
        # 2. Tabelas comparativas
        self.gerar_tabelas_comparativas()
        
        # 3. Gráficos profissionais
        self.gerar_graficos_profissionais()
        
        # 4. PDF consolidado
        self.gerar_pdf_consolidado()
        
        print(f"\n{'='*100}")
        print(f"{'✓ RELATÓRIO COMPLETO GERADO COM SUCESSO':^100}")
        print(f"{'='*100}\n")
        
        # Sumário final
        print(f"\n{'SUMÁRIO DE ARQUIVOS GERADOS':^100}")
        print(f"{'-'*100}")
        
        n_csvs = len(list((self.pasta_saida / 'CSVs').glob('*.csv')))
        n_latex = len(list((self.pasta_saida / 'Tabelas_LaTeX').glob('*.tex')))
        n_mrt = len(list((self.pasta_saida / 'Graficos' / 'MRT').glob('*.png')))
        n_t2f = len(list((self.pasta_saida / 'Graficos' / 'T2F').glob('*.png')))
        n_comp = len(list((self.pasta_saida / 'Graficos' / 'Comparacoes').glob('*.png')))
        n_pdf = len(list((self.pasta_saida / 'PDFs').glob('*.pdf')))
        
        print(f"  📄 CSVs:              {n_csvs:3d} arquivos")
        print(f"  📋 Tabelas LaTeX:     {n_latex:3d} arquivos")
        print(f"  📊 Gráficos MRT:      {n_mrt:3d} imagens")
        print(f"  📊 Gráficos T2F:      {n_t2f:3d} imagens")
        print(f"  📊 Comparações:       {n_comp:3d} imagens")
        print(f"  📕 PDFs:              {n_pdf:3d} arquivo(s)")
        print(f"{'-'*100}")
        print(f"  📁 Pasta de saída: {self.pasta_saida}")
        print(f"{'='*100}\n")
# ==============================================================================
# 3. EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    
    # Configuração
    pasta_entrada = Path("C:/Users/Leonardo Felipe/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_10/Processados_HDF5")
    pasta_saida = "Resultados_Cap4_Tese_Final"
    
    # Cria analisador
    analisador = AnalisadorCapitulo4(pasta_entrada, pasta_saida)
    
    # Processa todos os casos
    df = analisador.processar_todos()
    
    # Gera relatório completo
    if len(df) > 0:
        analisador.gerar_relatorio_completo()
        
        print("\n" + "="*100)
        print(f"{'ESTRUTURA FINAL DE DIRETÓRIOS':^100}")
        print("="*100)
        print(f"\n{pasta_saida}/")
        print("  ├── 📁 CSVs/")
        print("  │   ├── 00_RESUMO_GERAL_TODOS_CASOS.csv")
        print("  │   ├── Tabela_1_Resumo_Geral.csv")
        print("  │   └── Tabela_2_Comparacao_MRT_vs_T2F.csv")
        print("  ├── 📁 Tabelas_LaTeX/")
        print("  │   ├── Tabela_1_Resumo_Geral.tex")
        print("  │   └── Tabela_2_Comparacao_MRT_vs_T2F.tex")
        print("  ├── 📁 Graficos/")
        print("  │   ├── 📁 MRT/")
        print("  │   │   ├── MRT_A-G_COM_Aterramento_Completo.png")
        print("  │   │   └── MRT_A-G_SEM_Aterramento_Completo.png")
        print("  │   ├── 📁 T2F/")
        print("  │   │   ├── T2F_ABC_COM_Aterramento_Completo.png")
        print("  │   │   ├── T2F_AB_COM_Aterramento_Completo.png")
        print("  │   │   ├── T2F_AC_COM_Aterramento_Completo.png")
        print("  │   │   ├── T2F_BC_COM_Aterramento_Completo.png")
        print("  │   │   └── ...")
        print("  │   └── 📁 Comparacoes/")
        print("  │       ├── Comparacao_MRT_vs_T2F_COM_Aterramento.png")
        print("  │       └── Comparacao_MRT_vs_T2F_SEM_Aterramento.png")
        print("  └── 📁 PDFs/")
        print("      └── Relatorio_Completo_Cap4.pdf")
        print("\n" + "="*100)
        print(f"{'✓ ANÁLISE CONCLUÍDA - PRONTO PARA TESE!':^100}")
        print("="*100 + "\n")
    else:
        print("\n✗ Nenhum dado processado. Verifique os arquivos de entrada.\n")
