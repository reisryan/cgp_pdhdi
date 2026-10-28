import matplotlib as plt
import numpy as np
import sys

def grafico(hist_pai, hist_lambda, nome_arquivo):
    plt.figure(figsize=(10, 5))
    plt.plot(hist_pai, label='Melhor indivíduo da geração', color='blue')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Evolução do Melhor Indivíduo') ## grafico com todas as n iteraçoes, e pegar a media de cada ponto de cada iteraçao
    plt.legend()
    plt.grid()
    plt.savefig(f"{nome_arquivo}.png")
    plt.close()
    

    dados = np.array(hist_lambda, dtype=float)
    ger = np.arange(1, dados.shape[0] + 1)

    media = dados.mean(axis=1)
    melhor = dados.max(axis=1)       # menor MAE = melhor
    pior = dados.min(axis=1)

    plt.figure(figsize=(10, 5))
    plt.fill_between(ger, melhor, pior, alpha=0.2, label="mín–máx dos filhos")
    plt.plot(ger, media, label="média dos filhos")
    plt.plot(ger, melhor, "--", label="melhor filho")
    if hist_pai is not None:
        plt.plot(ger, hist_pai[:len(ger)], "k", linewidth=2, label="pai selecionado")
    plt.xlabel("Geração")
    plt.ylabel("Fitness (MAE)")
    plt.title("Fitness dos filhos por geração")
    plt.legend()
    plt.grid()
    plt.savefig(f"{nome_arquivo}_filhos.png", dpi=200, bbox_inches="tight")
    plt.close()

def main():
    if len(sys.argv) < 3:
        print("Uso: python3 data_analysis.py graphic <hist_file> or \n python3 data_analysis.py graph <genoma> or \n or python3 data_analysis.py images [sequence of functions] image gtpath")

    if sys.argv[1] == "graphic":
        hist_file = sys.argv[2]
        hist_pai, hist_lambda = [], []
        lines = open(hist_file, "r").readlines()
        nome_arquivo = hist_file.split(".")[0]
            
        grafico(hist_pai, hist_lambda, nome_arquivo)

if __name__ == "__main__":
    main()