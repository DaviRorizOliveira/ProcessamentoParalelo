import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Define o nome do arquivo de resultados
nomeArquivo = 'resultados.csv'
nomeArquivo2 = 'resultadosT2.csv'

def configPlotBase(titulo, ylabel, xlabel='Tamanho da Matriz (N)'):
    """Configuração padrão de estilo para todos os gráficos"""
    plt.figure(figsize=(12, 7))
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.style.use('seaborn-v0_8-whitegrid') # Estilo limpo

def salvarGrafico(nome_arquivo):
    os.makedirs("graficos", exist_ok=True)
    caminho = os.path.join("graficos", nome_arquivo)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo em: {caminho}")
    plt.close()

def plotarGraficos(dfTempo, dfSpeedup, dfEficiencia):
    """
    Recebe os 3 DataFrames processados e gera os gráficos comparativos.
    """
    # Lista de configurações de cores e marcadores para manter consistência
    
    # Dicionário: 'Identificador': ('Cor', 'Marcador', 'Legenda')
    config_visual = {
        'Seq':      ('#000000', 'o', 'Sequencial'),
        'OMP_2':    ('#ffcc80', 's', 'OMP (2 threads)'),
        'OMP_4':    ('#ff9800', '^', 'OMP (4 threads)'),
        'OMP_8':    ('#e65100', 'd', 'OMP (8 threads)'),
        'MPI_2':    ('#a5d6a7', 'v', 'MPI (2 procs)'),
        'MPI_4':    ('#4caf50', '<', 'MPI (4 procs)'),
        'MPI_8':    ('#1b5e20', '>', 'MPI (8 procs)'),
        'CUDA_N':   ('#ce93d8', 'x', 'CUDA Naive'),
        'CUDA_T':   ('#7b1fa2', '*', 'CUDA Tiled')
    }

    tamanhos = dfTempo['tamMatriz']

    # ==========================================
    # 1. GRÁFICO DE TEMPO (Escala Log-Log)
    # ==========================================
    configPlotBase('Comparação de Tempo de Execução', 'Tempo (segundos)')
    
    # Plota as linhas (Verifica se a coluna existe antes de plotar)
    if 'tempoSequencial' in dfTempo: 
        plt.plot(tamanhos, dfTempo['tempoSequencial'], label=config_visual['Seq'][2], color=config_visual['Seq'][0], marker=config_visual['Seq'][1], lw=2)
    if 'tempo2Thread' in dfTempo :
        plt.plot(tamanhos, dfTempo['tempo2Thread'],    label=config_visual['OMP_2'][2], color=config_visual['OMP_2'][0], marker=config_visual['OMP_2'][1])
    if 'tempo4Thread' in dfTempo :
        plt.plot(tamanhos, dfTempo['tempo4Thread'],    label=config_visual['OMP_4'][2], color=config_visual['OMP_4'][0], marker=config_visual['OMP_4'][1])
    if 'tempo8Thread' in dfTempo:    
        plt.plot(tamanhos, dfTempo['tempo8Thread'],    label=config_visual['OMP_8'][2], color=config_visual['OMP_8'][0], marker=config_visual['OMP_8'][1])
    
    if 'tempoMPI2' in dfTempo:       
        plt.plot(tamanhos, dfTempo['tempoMPI2'],       label=config_visual['MPI_2'][2], color=config_visual['MPI_2'][0], marker=config_visual['MPI_2'][1], ls='--')
    if 'tempoMPI4' in dfTempo:       
        plt.plot(tamanhos, dfTempo['tempoMPI4'],       label=config_visual['MPI_4'][2], color=config_visual['MPI_4'][0], marker=config_visual['MPI_4'][1], ls='--')
    if 'tempoMPI8' in dfTempo:       
        plt.plot(tamanhos, dfTempo['tempoMPI8'],       label=config_visual['MPI_8'][2], color=config_visual['MPI_8'][0], marker=config_visual['MPI_8'][1], ls='--')
    
    if 'tempoCUDA_naive' in dfTempo: 
        plt.plot(tamanhos, dfTempo['tempoCUDA_naive'], label=config_visual['CUDA_N'][2], color=config_visual['CUDA_N'][0], marker=config_visual['CUDA_N'][1])
    if 'tempoCUDA_tiled' in dfTempo: 
        plt.plot(tamanhos, dfTempo['tempoCUDA_tiled'], label=config_visual['CUDA_T'][2], color=config_visual['CUDA_T'][0], marker=config_visual['CUDA_T'][1], lw=2)

    # Configurações de Eixo
    plt.xscale('log', base=2) # X em Log base 2 (128, 256, 512...)
    plt.yscale('log')         # Y em Log (essencial para ver CUDA vs Seq)
    plt.xticks(tamanhos, labels=tamanhos.astype(str)) # Força labels corretos
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Legenda fora para não cobrir dados
    
    salvarGrafico('01_tempo_execucao.png')

    # ==========================================
    # 2. GRÁFICO DE SPEEDUP
    # ==========================================
    configPlotBase('Speedup (vs Sequencial)', 'Speedup (x vezes mais rápido)')

    # Linha de referência
    plt.axhline(y=1, color='red', linestyle=':', label='Base Sequencial (1.0)')

    if 'speedup2Thread' in dfSpeedup:    
        plt.plot(tamanhos, dfSpeedup['speedup2Thread'],    label=config_visual['OMP_2'][2], color=config_visual['OMP_2'][0], marker=config_visual['OMP_2'][1])
    if 'speedup4Thread' in dfSpeedup:    
        plt.plot(tamanhos, dfSpeedup['speedup4Thread'],    label=config_visual['OMP_4'][2], color=config_visual['OMP_4'][0], marker=config_visual['OMP_4'][1])
    if 'speedup8Thread' in dfSpeedup:    
        plt.plot(tamanhos, dfSpeedup['speedup8Thread'],    label=config_visual['OMP_8'][2], color=config_visual['OMP_8'][0], marker=config_visual['OMP_8'][1])
    
    if 'speedupMPI2' in dfSpeedup:       
        plt.plot(tamanhos, dfSpeedup['speedupMPI2'],       label=config_visual['MPI_2'][2], color=config_visual['MPI_2'][0], marker=config_visual['MPI_2'][1], ls='--')
    if 'speedupMPI4' in dfSpeedup:       
        plt.plot(tamanhos, dfSpeedup['speedupMPI4'],       label=config_visual['MPI_4'][2], color=config_visual['MPI_4'][0], marker=config_visual['MPI_4'][1], ls='--')
    if 'speedupMPI8' in dfSpeedup:       
        plt.plot(tamanhos, dfSpeedup['speedupMPI8'],       label=config_visual['MPI_8'][2], color=config_visual['MPI_8'][0], marker=config_visual['MPI_8'][1], ls='--')

    if 'speedupCUDA_naive' in dfSpeedup: 
        plt.plot(tamanhos, dfSpeedup['speedupCUDA_naive'], label=config_visual['CUDA_N'][2], color=config_visual['CUDA_N'][0], marker=config_visual['CUDA_N'][1])
    if 'speedupCUDA_tiled' in dfSpeedup: 
        plt.plot(tamanhos, dfSpeedup['speedupCUDA_tiled'], label=config_visual['CUDA_T'][2], color=config_visual['CUDA_T'][0], marker=config_visual['CUDA_T'][1], lw=2)

    plt.xscale('log', base=2)
    # plt.yscale('log') # Opcional: Ative se o Speedup do CUDA for > 100x e esconder os outros
    plt.xticks(tamanhos, labels=tamanhos.astype(str))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    salvarGrafico('02_speedup.png')

    # ==========================================
    # 3. GRÁFICO DE EFICIÊNCIA
    # ==========================================
    configPlotBase('Eficiência Paralela', 'Eficiência (Speedup / N_Procs)')
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Eficiência Ideal (1.0)')

    if 'eficiencia2Thread' in dfEficiencia: 
        plt.plot(tamanhos, dfEficiencia['eficiencia2Thread'], label=config_visual['OMP_2'][2], color=config_visual['OMP_2'][0], marker=config_visual['OMP_2'][1])
    if 'eficiencia4Thread' in dfEficiencia: 
        plt.plot(tamanhos, dfEficiencia['eficiencia4Thread'], label=config_visual['OMP_4'][2], color=config_visual['OMP_4'][0], marker=config_visual['OMP_4'][1])
    if 'eficiencia8Thread' in dfEficiencia: 
        plt.plot(tamanhos, dfEficiencia['eficiencia8Thread'], label=config_visual['OMP_8'][2], color=config_visual['OMP_8'][0], marker=config_visual['OMP_8'][1])
    
    if 'eficienciaMPI2' in dfEficiencia:    
        plt.plot(tamanhos, dfEficiencia['eficienciaMPI2'],    label=config_visual['MPI_2'][2], color=config_visual['MPI_2'][0], marker=config_visual['MPI_2'][1], ls='--')
    if 'eficienciaMPI4' in dfEficiencia:    
        plt.plot(tamanhos, dfEficiencia['eficienciaMPI4'],    label=config_visual['MPI_4'][2], color=config_visual['MPI_4'][0], marker=config_visual['MPI_4'][1], ls='--')
    if 'eficienciaMPI8' in dfEficiencia:    
        plt.plot(tamanhos, dfEficiencia['eficienciaMPI8'],    label=config_visual['MPI_8'][2], color=config_visual['MPI_8'][0], marker=config_visual['MPI_8'][1], ls='--')

    # CUDA geralmente não se plota eficiência clássica (Speedup/Cores) da mesma forma que CPU, 
    # Geralmente a Eficiência > 1 em GPU é comum devido à arquitetura.

    plt.xscale('log', base=2)
    plt.ylim(0, 1.2) # Foca na faixa de 0 a 120%
    plt.xticks(tamanhos, labels=tamanhos.astype(str))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    salvarGrafico('03_eficiencia.png')



def processarDados():
    """
    Lê o CSV, e formata os dados para escrever as tabelas.
    """
    try:
        # Carrega o arquivo CSV
        df1 = pd.read_csv(nomeArquivo)
        
        df2Raw = pd.read_csv(nomeArquivo2)

        mpiCols = ['tamMatriz','tempoMPI','speedupMPI','eficienciaMPI']

        dfTemp = df2Raw.loc[:,mpiCols]

        df2Grouped = dfTemp.groupby('tamMatriz')

        dfMPInp2 = df2Grouped.nth(0).reset_index().rename(columns={
            'tempoMPI': 'tempoMPI2',
            'speedupMPI': 'speedupMPI2',
            'eficienciaMPI': 'eficienciaMPI2'
        })

        dfMPInp4 = df2Grouped.nth(1).reset_index().rename(columns={
            'tempoMPI': 'tempoMPI4',
            'speedupMPI': 'speedupMPI4',
            'eficienciaMPI': 'eficienciaMPI4'
        })

        dfMPInp8 = df2Grouped.nth(2).reset_index().rename(columns={
            'tempoMPI': 'tempoMPI8',
            'speedupMPI': 'speedupMPI8',
            'eficienciaMPI': 'eficienciaMPI8'
        })
        
        df = pd.merge(df1, dfMPInp2, on='tamMatriz')
        df = pd.merge(df,dfMPInp4, on='tamMatriz')
        df = pd.merge(df,dfMPInp8, on='tamMatriz')

        dfTempo = df.loc[:,[
            'tamMatriz',
            'tempoSequencial',
            'tempo2Thread','tempo4Thread','tempo8Thread',
            'tempoMPI2','tempoMPI4','tempoMPI8',
            'tempoCUDA_naive','tempoCUDA_tiled'
        ]].copy()
        dfSpeedup = df.loc[:,[
            'tamMatriz',
            'speedup2Thread','speedup4Thread','speedup8Thread',
            'speedupMPI2','speedupMPI4','speedupMPI8',
            'speedupCUDA_naive','speedupCUDA_tiled'
        ]].copy()
        dfEficiencia = df.loc[:,[
            'tamMatriz',
            'eficiencia2Thread','eficiencia4Thread','eficiencia8Thread',
            'eficienciaMPI2','eficienciaMPI4','eficienciaMPI8',
        ]].copy()

    except Exception as e:
        print(f"Na função processar dados, ocorreu um erro inesperado: {e}")
        return None,None,None

    return dfTempo, dfSpeedup,dfEficiencia

def escreveTabelas(dfTempo,dfSpeedup,dfEficiencia):
    """
    Recebe as listas dos DataFrames e os salva em um arquivo de texto.
    """
    if dfTempo is None : 
        return

    listDf = [("Tabela de TEMPO", dfTempo), 
              ("Tabela de SPEEDUP", dfSpeedup), 
              ("Tabela de EFICIENCIA", dfEficiencia)]

    # Garante que o diretório 'tabelas' exista
    os.makedirs("tabelas", exist_ok=True)

    with open("tabelas/tabelas.txt", 'w') as arq :
        arq.write("Analise Comparativa (Sequencial vs OMP vs MPI vs CUDA)\n\n")

        for title, df in listDf :
            arq.write("="*60 + "\n")
            arq.write(f"{title}\n")
            arq.write("="*60 + "\n")

            tabelaString = df.set_index('tamMatriz').to_string(float_format='{:.6f}'.format)

            arq.write(tabelaString)
            arq.write("\n\n")
    print("Tabela de resultados foi salva em 'tabelas/tabelas.txt'")
def analiseDados():
    """
    Função principal de analise
    """
    t, s, e = processarDados()

    if t is not None :
        escreveTabelas(t,s,e)
        plotarGraficos(t,s,e)

def main ():
    analiseDados()

if __name__ == "__main__":
    main()