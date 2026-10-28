from deap import base, creator, tools

import numpy as np
import matplotlib
import random
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import functions_pi as fpi
from cgp import CGP, Imagem, ImagemBinaria
from cgpExecutor import CGPExecutor
from cgpEvolve import CGPEvolve

np.random.seed(1)

operadores = {
    1: {"funcao": fpi.clahe, "entrada": Imagem, "saida": Imagem},
    2: {"funcao": fpi.otsu, "entrada": Imagem, "saida": ImagemBinaria},
    3: {"funcao": fpi.multiotsu, "entrada": Imagem, "saida": ImagemBinaria},
    4: {"funcao": fpi.nick, "entrada": Imagem, "saida": ImagemBinaria},
    5: {"funcao": fpi.nick, "entrada": ImagemBinaria, "saida": ImagemBinaria},
    6: {"funcao": fpi.sauvola, "entrada": Imagem, "saida": ImagemBinaria},
    7: {"funcao": fpi.sauvola, "entrada": ImagemBinaria, "saida": ImagemBinaria},
    8: {"funcao": fpi.open_morphology, "entrada": ImagemBinaria,"saida": ImagemBinaria},
    9: {"funcao": fpi.close_morphology, "entrada": ImagemBinaria, "saida": ImagemBinaria},
    10: {"funcao": fpi.square_root, "entrada": Imagem, "saida": Imagem},
    11: {"funcao": fpi.square, "entrada": Imagem, "saida": Imagem},
    12: {"funcao": fpi.f_exp, "entrada": Imagem, "saida": Imagem},
    13: {"funcao": fpi.f_log, "entrada": Imagem, "saida": Imagem},
    14: {"funcao": fpi.median_blur, "entrada": Imagem, "saida": Imagem},
    15: {"funcao": fpi.gaussian_blur, "entrada": Imagem, "saida": Imagem},
    16: {"funcao": fpi.laplacian, "entrada": Imagem, "saida": Imagem},
    17: {"funcao": fpi.sobel, "entrada": Imagem, "saida": Imagem},
    18: {"funcao": fpi.erode, "entrada": ImagemBinaria, "saida": ImagemBinaria},
    19: {"funcao": fpi.dilate, "entrada": ImagemBinaria, "saida": ImagemBinaria},
    20: {"funcao": fpi.normalize, "entrada": Imagem, "saida": Imagem},
    21: {"funcao": fpi.f_invert, "entrada": ImagemBinaria, "saida": ImagemBinaria},
    22: {"funcao": fpi.bilateral_filter, "entrada": Imagem, "saida": Imagem},
    23: {"funcao": fpi.morph_gradient, "entrada": Imagem, "saida": Imagem},
    24: {"funcao": fpi.f_sharpen, "entrada": Imagem, "saida": Imagem},
    25: {"funcao": fpi.TSMO, "entrada": Imagem, "saida": ImagemBinaria}
    # adição de 2 imagens
    # subtração de 2 imagens
}

# CGP
cgp = CGP(1, operadores, 30, 1, 2, 5)
executor = CGPExecutor(cgp)
evolve = CGPEvolve(cgp, executor)

# DEAP
creator.create("FitnessMax", base.Fitness,  weights=(1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individuo, cgp.gera_genoma)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mutate", evolve.mutacao)

input, output = [], []
pasta = 'OriginalImages'


def avaliar_imagem(imagepath, outputpath, n_geracoes=50, n_lambda=5):
    toolbox.register("evaluate", evolve.avaliar, inputs=[imagepath], outputs=[outputpath])
    pai = toolbox.individual()
    pai.fitness.values = toolbox.evaluate(pai)

    hist_geracao, hist_pai, hist_lambda = [], [], []
    for geracao in range(1, n_geracoes+1):
        filhos = []
        for l in range(n_lambda):
            filho, = toolbox.mutate(pai)
            filho.fitness.values = toolbox.evaluate(filho)
            filhos.append(filho)

        melhor = tools.selBest(filhos, 1)[0]
        if melhor.fitness.values[0] >= pai.fitness.values[0]: # mse <=, pnsr >=
            neutros = [f for f in filhos if f.fitness.values[0] == pai.fitness.values[0]]
            pai = random.choice(neutros) if neutros else melhor

        print(f"Geração {geracao + 1}: Melhor fitness = {pai.fitness.values[0]}")
        print(f"Média fitness = {sum([f.fitness.values[0] for f in filhos]) / len(filhos)}")
        hist_pai.append(pai.fitness.values[0])
        hist_lambda.append([f.fitness.values[0] for f in filhos])

        if geracao != 0 and geracao % 10 == 0:
            hist_geracao.append(hist_pai)
            nome_arquivo = f"{os.path.basename(imagepath).split('.')[0]}_geracao_{geracao + 1}"
            grafico(hist_pai, hist_lambda, nome_arquivo)

    return pai.fitness.values[0], pai, hist_pai, hist_lambda

def grafico(hist_pai, hist_lambda, nome_arquivo):
    plt.figure(figsize=(10, 5))
    plt.plot(hist_pai, label='Melhor indivíduo da geração', color='blue')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Evolução do Melhor Indivíduo')
    plt.legend()
    plt.grid()
    plt.savefig(f"{nome_arquivo}.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(len(hist_lambda[0])):
        plt.plot([hist_lambda[j][i] for j in range(len(hist_lambda))], label=f'Filho {i + 1}')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Evolução dos Filhos da Geração')
    plt.legend()
    plt.grid()
    plt.savefig(f"{nome_arquivo}_filhos.png")
    plt.close()

#som = 0
f, p, h, hl = avaliar_imagem('OriginalImages/HW01_dibco2013.bmp', 'GTimages/HW01_estGT_dibco2013.tiff', n_geracoes=50, n_lambda=5)
"""som += f
f, p, h, hl = avaliar_imagem('OriginalImages/HW02_dibco2013.bmp', 'GTimages/HW02_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/HW03_dibco2013.bmp', 'GTimages/HW03_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/HW04_dibco2013.bmp', 'GTimages/HW04_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/HW05_dibco2013.bmp', 'GTimages/HW05_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/HW06_dibco2013.tiff', 'GTimages/HW06_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/HW07_dibco2013.bmp', 'GTimages/HW07_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/HW08_dibco2013.tiff', 'GTimages/HW08_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/PR01_dibco2013.bmp', 'GTimages/PR01_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/PR02_dibco2013.bmp', 'GTimages/PR02_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/PR03_dibco2013.bmp', 'GTimages/PR03_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/PR04_dibco2013.bmp', 'GTimages/PR04_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/PR05_dibco2013.bmp', 'GTimages/PR05_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/PR06_dibco2013.bmp', 'GTimages/PR06_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/PR07_dibco2013.bmp', 'GTimages/PR07_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
f, p, h, hl = avaliar_imagem('OriginalImages/PR08_dibco2013.bmp', 'GTimages/PR08_estGT_dibco2013.tiff', n_geracoes=100, n_lambda=5)
som += f
print(f"Somatório dos resultados: {som}")
print(f"Média dos resultados: {som/16}")
grafico(h, hl, 'PR08_dibco2013')"""