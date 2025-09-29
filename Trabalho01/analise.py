import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analiseDados():
    df = Tabelas()
    Graficos(df)

def escreveTabelas(vetorDf) :
    with open("tabelas/tabelas.txt", 'w') as arq :
        for x in range(len(vetorDf)) :
            arq.write(f"{vetorDf[x]}\n\n")

def Tabelas() :
    df = pd.read_csv("resultados.csv")
    dfTempo = df.loc[:,['tamMatriz','tempoSequencial','tempo2Thread','tempo4Thread','tempo8Thread']]
    dfSpeedup = df.loc[:,['tamMatriz','speedup2Thread','speedup4Thread','speedup8Thread']]
    dfEficiencia = df.loc[:,['tamMatriz','eficiencia2Thread','eficiencia4Thread','eficiencia8Thread']]

    vetorDf = [dfTempo,dfSpeedup,dfEficiencia]

    escreveTabelas(vetorDf)

    return vetorDf

def Graficos(vetorDf) :

    dfTempo,dfSpeedup,dfEficiencia = vetorDf

    plt.style.use('seaborn-v0_8')
    plt.figure(figsize = (18,12))

# ============Tempos de Execução============
    plt.figure(figsize = (18,12))
    plt.plot(dfTempo['tamMatriz'],dfTempo['tempoSequencial'], 'o-',linewidth=2, markersize=8, label='Sequencial', color='#1f77b4')
    plt.plot(dfTempo['tamMatriz'], dfTempo['tempo2Thread'], 's-', markersize=6, label='2 Threads', color='#ff7f0e')
    plt.plot(dfTempo['tamMatriz'], dfTempo['tempo4Thread'], '^-', markersize=6, label='4 Threads', color='#2ca02c')
    plt.plot(dfTempo['tamMatriz'], dfTempo['tempo8Thread'], 'd-', markersize=6, label='8 Threads', color='#d62728')

    plt.xticks(dfTempo['tamMatriz'], [f"{int(x)}" for x in dfTempo['tamMatriz']])
    
    x = []
    termoatual = 0
    razao = 0.5
    limite = 10
    while termoatual <= limite:
        x.append(termoatual)
        termoatual += razao

    plt.yticks(x)

    plt.xlabel('Tamanho da Matriz', fontsize=12, fontweight='bold')
    plt.ylabel('Tempo (segundos)', fontsize=12, fontweight='bold')
    plt.title('Tempo de Execução vs Tamanho da Matriz', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('graficos/graficoExecucao_dgemm.png', dpi=300, bbox_inches='tight')
    plt.close()
 # ========== GRÁFICO 2: SPEEDUP ==========
    plt.subplot(2, 2, 2)
    plt.plot(dfSpeedup['tamMatriz'], dfSpeedup['speedup2Thread'], 's-', markersize=6, label='2 Threads', color='#ff7f0e')
    plt.plot(dfSpeedup['tamMatriz'], dfSpeedup['speedup4Thread'], '^-', markersize=6, label='4 Threads', color='#2ca02c')
    plt.plot(dfSpeedup['tamMatriz'], dfSpeedup['speedup8Thread'], 'd-', markersize=6, label='8 Threads', color='#d62728')

    plt.xticks(dfSpeedup['tamMatriz'], [f"{int(x)}" for x in dfSpeedup['tamMatriz']])
    
    # Linhas de referência para speedup ideal
    plt.axhline(y=2, color='#ff7f0e', linestyle='--', alpha=0.5, label='Ideal 2T')
    plt.axhline(y=4, color='#2ca02c', linestyle='--', alpha=0.5, label='Ideal 4T')
    plt.axhline(y=8, color='#d62728', linestyle='--', alpha=0.5, label='Ideal 8T')
    
    plt.xlabel('Tamanho da Matriz', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup', fontsize=12, fontweight='bold')
    plt.title('Speedup vs Tamanho da Matriz', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('graficos/graficoSpeedup_dgemm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== GRÁFICO 3: EFICIÊNCIA ==========
    plt.figure(figsize=(10, 6))
    plt.plot(dfEficiencia['tamMatriz'], dfEficiencia['eficiencia2Thread'], 's-', markersize=6, label='2 Threads', color='#ff7f0e')
    plt.plot(dfEficiencia['tamMatriz'], dfEficiencia['eficiencia4Thread'], '^-', markersize=6, label='4 Threads', color='#2ca02c')
    plt.plot(dfEficiencia['tamMatriz'], dfEficiencia['eficiencia8Thread'], 'd-', markersize=6, label='8 Threads', color='#d62728')

    plt.xticks(dfEficiencia['tamMatriz'], [f"{int(x)}" for x in dfEficiencia['tamMatriz']])
    
    # Linha de referência para eficiência ideal (100%)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Eficiência Ideal (100%)')
    
    plt.xlabel('Tamanho da Matriz', fontsize=12, fontweight='bold')
    plt.ylabel('Eficiência', fontsize=12, fontweight='bold')
    plt.title('Eficiência vs Tamanho da Matriz', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graficos/graficoEficiencia_dgemm.png', dpi=300, bbox_inches='tight')
    plt.close()

def main ():
    analiseDados()

if __name__ == "__main__":
    main()



