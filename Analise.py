"""
================================================================================
ANÁLISE CAPÍTULO 4 - ORGANIZADO POR TIPO DE CURTO-CIRCUITO
Salva tudo estruturado: CSVs, Gráficos e PDFs separados por tipo de falta
================================================================================
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ==============================================================================
# PROCESSADOR (MESMO DE ANTES)
# ==============================================================================
class ProcessadorSinais:
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
                    raise ValueError("Sem 't'")
                
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
                            'raw': dados_raw, 'rms': None, 'seq': None, 'thd': None
                        }
            
            print(f"✓ {Path(self.arquivo).stem[:60]}")
        except Exception as e:
            raise Exception(f"Erro: {e}")
    
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
# ANALISADOR COM SALVAMENTO ORGANIZADO POR TIPO DE FALTA
# ==============================================================================
class AnalisadorCapitulo4:
    def __init__(self, pasta_entrada, pasta_saida='Resultados_Cap4'):
        self.pasta = Path(pasta_entrada)
        self.pasta_saida = Path(pasta_saida)
        self.arquivos = sorted(self.pasta.glob('*_py.mat'))
        self.resultados = []
        self.df = None
        
        # Cria pastas
        (self.pasta_saida / 'CSVs').mkdir(parents=True, exist_ok=True)
        (self.pasta_saida / 'Graficos').mkdir(parents=True, exist_ok=True)
        (self.pasta_saida / 'PDFs').mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"ANALISADOR CAP. 4 - Organizado por Tipo de Falta")
        print(f"{'='*80}")
        print(f"Entrada: {self.pasta}")
        print(f"Saída: {self.pasta_saida}")
        print(f"Arquivos: {len(self.arquivos)}")
        print(f"{'='*80}\n")
    
    def processar_todos(self):
        print("PROCESSANDO...\n")
        
        for i, arq in enumerate(self.arquivos, 1):
            print(f"[{i}/{len(self.arquivos)}] ", end='')
            try:
                proc = ProcessadorSinais(str(arq))
                self.resultados.append(self._extrair_metricas(proc, arq.stem))
            except Exception as e:
                print(f"✗ {e}")
        
        if len(self.resultados) > 0:
            self.df = pd.DataFrame(self.resultados)
            print(f"\n{'='*80}")
            print(f"✓ {len(self.df)} casos processados")
            print(f"{'='*80}\n")
        else:
            self.df = pd.DataFrame()
        
        return self.df
    
    def _extrair_metricas(self, proc, nome_arq):
        partes = nome_arq.split('__')
        simulacao = partes[0]
        caso = partes[1] if len(partes) > 1 else ''
        
        tipo_sistema = 'T2F' if 'Qualificacao' in simulacao else 'MRT'
        sem_aterramento = 'sem_terra' in simulacao or 'no_Ground' in simulacao
        
        # Identifica tipo de falta
        if 'Sem_Falta' in caso:
            tipo_falta = 'Normal'
        elif '_ABC' in caso:
            tipo_falta = 'ABC'
        elif '_AB' in caso:
            tipo_falta = 'AB'
        elif '_AC' in caso:
            tipo_falta = 'AC'
        elif '_BC' in caso:
            tipo_falta = 'BC'
        elif 'Falta_A' in caso:
            tipo_falta = 'A-G'
        else:
            tipo_falta = 'Desconhecido'
        
        idx_falta = proc.detectar_falta('I_822')
        
        resultado = {
            'arquivo': nome_arq,
            'tipo_sistema': tipo_sistema,
            'sem_aterramento': sem_aterramento,
            'tipo_falta': tipo_falta,
            'm1': float(proc.m1),
            't_falta': float(proc.t[idx_falta]),
            'processador': proc
        }
        
        # Métricas pontos 800, 816, 818, 820, 822
        for ponto in ['800', '816', '818', '820', '822']:
            nome_i = f'I_{ponto}'
            if nome_i in proc.dados and proc.dados[nome_i]['rms'] is not None:
                rms = proc.dados[nome_i]['rms']
                resultado[f'{ponto}_I_pico_max'] = float(np.max(rms))
                
                if proc.dados[nome_i]['seq']:
                    seq = proc.dados[nome_i]['seq']
                    resultado[f'{ponto}_I0'] = float(seq['I0'][idx_falta])
                    resultado[f'{ponto}_I1'] = float(seq['I1'][idx_falta])
                    resultado[f'{ponto}_I2'] = float(seq['I2'][idx_falta])
        
        return resultado
    
    def salvar_por_tipo_falta(self):
        """SALVA TUDO ORGANIZADO POR TIPO DE FALTA"""
        
        if self.df is None or len(self.df) == 0:
            return
        
        print(f"\n{'='*80}")
        print("SALVANDO POR TIPO DE FALTA")
        print(f"{'='*80}\n")
        
        for tipo_falta in self.df['tipo_falta'].unique():
            if tipo_falta == 'Desconhecido':
                continue
            
            print(f"--- {tipo_falta} ---")
            df_falta = self.df[self.df['tipo_falta'] == tipo_falta]
            
            # 1. CSV Resumo
            colunas = [c for c in df_falta.columns if c != 'processador']
            csv_nome = f'Falta_{tipo_falta}_Resumo.csv'
            df_falta[colunas].to_csv(self.pasta_saida / 'CSVs' / csv_nome, index=False, encoding='utf-8-sig')
            print(f"  ✓ CSV: {csv_nome}")
            
            # 2. Comparações T2F vs MRT
            for com_terra in [True, False]:
                df_sub = df_falta[df_falta['sem_aterramento'] == (not com_terra)]
                
                t2f = df_sub[df_sub['tipo_sistema'] == 'T2F']
                mrt = df_sub[df_sub['tipo_sistema'] == 'MRT']
                
                if len(t2f) > 0 and len(mrt) > 0:
                    terra_label = 'COM_Terra' if com_terra else 'SEM_Terra'
                    
                    # CSV Comparação
                    comp_data = []
                    for col in ['822_I_pico_max', '822_I0', '822_I1', '822_I2']:
                        if col in t2f.columns and col in mrt.columns:
                            val_t2f = t2f[col].mean()
                            val_mrt = mrt[col].mean()
                            delta = ((val_t2f - val_mrt) / val_mrt * 100) if val_mrt != 0 else 0
                            comp_data.append({'Metrica': col, 'T2F': val_t2f, 'MRT': val_mrt, 'Delta_%': delta})
                    
                    pd.DataFrame(comp_data).to_csv(
                        self.pasta_saida / 'CSVs' / f'Falta_{tipo_falta}_{terra_label}_Comparacao.csv',
                        index=False, encoding='utf-8-sig'
                    )
                    print(f"  ✓ Comparação: {terra_label}")
                    
                    # Gráfico
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    for sistema in ['T2F', 'MRT']:
                        df_s = df_sub[df_sub['tipo_sistema'] == sistema].sort_values('m1')
                        if len(df_s) > 0 and '822_I_pico_max' in df_s.columns:
                            ax.plot(df_s['m1']*100, df_s['822_I_pico_max'], 'o-', 
                                   label=sistema, linewidth=2, markersize=8)
                    
                    ax.set_xlabel('Posição (%)', fontsize=12)
                    ax.set_ylabel('I_pico (A)', fontsize=12)
                    ax.set_title(f'Falta {tipo_falta} - {terra_label}', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    fig_nome = f'Falta_{tipo_falta}_{terra_label}.png'
                    plt.savefig(self.pasta_saida / 'Graficos' / fig_nome, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  ✓ Gráfico: {fig_nome}")
        
        # PDF consolidado
        self._gerar_pdf()
        
        print(f"\n{'='*80}")
        print("✓ TODOS OS ARQUIVOS SALVOS")
        print(f"{'='*80}\n")
    
    def _gerar_pdf(self):
        """Gera PDF com todos os gráficos"""
        imgs = sorted((self.pasta_saida / 'Graficos').glob('*.png'))
        
        if len(imgs) == 0:
            return
        
        with PdfPages(self.pasta_saida / 'PDFs' / 'Relatorio_Graficos.pdf') as pdf:
            for img in imgs:
                fig = plt.figure(figsize=(11, 8.5))
                plt.imshow(plt.imread(img))
                plt.axis('off')
                plt.tight_layout()
                pdf.savefig(fig, dpi=150)
                plt.close()
        
        print(f"\n  ✓ PDF: Relatorio_Graficos.pdf")
    
    def gerar_relatorio_completo(self):
        """Gera TUDO"""
        print(f"\n{'#'*80}")
        print("RELATÓRIO COMPLETO")
        print(f"{'#'*80}\n")
        
        # 1. CSV Geral
        colunas = [c for c in self.df.columns if c != 'processador']
        self.df[colunas].to_csv(
            self.pasta_saida / 'CSVs' / 'RESUMO_GERAL.csv',
            index=False, encoding='utf-8-sig'
        )
        print("✓ RESUMO_GERAL.csv")
        
        # 2. Por tipo de falta
        self.salvar_por_tipo_falta()
        
        print(f"\n{'#'*80}")
        print(f"✓ CONCLUÍDO - Pasta: {self.pasta_saida}")
        print(f"{'#'*80}\n")


# ==============================================================================
# EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    pasta_in = Path("C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_10/Processados_HDF5")
    pasta_out = "Resultados_Cap4_Por_Tipo_Falta"
    
    analisador = AnalisadorCapitulo4(pasta_in, pasta_out)
    df = analisador.processar_todos()
    
    if len(df) > 0:
        analisador.gerar_relatorio_completo()
        
        print("\n📁 ESTRUTURA DE SAÍDA:")
        print(f"  {pasta_out}/")
        print("    📁 CSVs/")
        print("       📄 RESUMO_GERAL.csv (todos os casos)")
        print("       📄 Falta_ABC_Resumo.csv")
        print("       📄 Falta_ABC_COM_Terra_Comparacao.csv")
        print("       📄 Falta_BC_Resumo.csv")
        print("       📄 ...")
        print("    📁 Graficos/")
        print("       🖼️  Falta_ABC_COM_Terra.png")
        print("       🖼️  Falta_BC_SEM_Terra.png")
        print("       🖼️  ...")
        print("    📁 PDFs/")
        print("       📕 Relatorio_Graficos.pdf\n")
