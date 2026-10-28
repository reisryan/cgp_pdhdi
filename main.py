from deap import base, creator, tools

import sys
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



def avaliar_imagem(imagepath, outputpath, n_geracoes, n_lambda, nos, levels_back, nos_por_camada):
    cgp = CGP(1, operadores, nos, 1, levels_back, nos_por_camada)
    executor = CGPExecutor(cgp)
    evolve = CGPEvolve(cgp, executor)
    creator.create("FitnessMax", base.Fitness,  weights=(1.0,))
    creator.create("Individuo", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individuo, cgp.gera_genoma)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", evolve.mutacao)
    toolbox.register("evaluate", evolve.avaliar, inputs=[imagepath], outputs=[outputpath])
    pai = toolbox.individual()
    pai.fitness.values = toolbox.evaluate(pai)

    hist_geracao, hist_pai, hist_lambda = [], [], []
    for geracao in range(1, n_geracoes+1):
        filhos, melhores = [], []
        for l in range(n_lambda):
            filho, = toolbox.mutate(pai)
            filho.fitness.values = toolbox.evaluate(filho)
            filhos.append(filho)
            if filho.fitness.values[0] > pai.fitness.values[0]:
                melhores.append(filho)
        neutros = [f for f in filhos if f.fitness.values[0] == pai.fitness.values[0]]


        if len(melhores) > 0:
            pai = random.choice(melhores)
        elif len(melhores) == 0 and len(neutros) > 0:
            pai = random.choice(neutros)

        print(f"Geração {geracao}: FITNESS, pai: {pai.fitness.values[0]}, filhos: {[int(f.fitness.values[0]) for f in filhos]}")
        print(f"Média fitness = {sum([f.fitness.values[0] for f in filhos]) / len(filhos)}")
        hist_pai.append(pai.fitness.values[0])
        hist_lambda.append([f.fitness.values[0] for f in filhos])

        if geracao != 0 and geracao % 10 == 0:
            hist_geracao.append([hist_pai, hist_lambda])
            
    return pai.fitness.values[0], pai, hist_pai, hist_lambda



def main():
    if len(sys.argv) < 8:
        print("Uso: python main.py <iteracoes> <geracoes> <lambda> <nos> <levels_back> <nos_por_camada> <imagepath> <gtpath>")
        sys.exit(1)
    args = sys.argv[1:]
    iter = int(args[0])
    geracoes = int(args[1])
    arglambda = int(args[2])
    nos = int(args[3])
    levels_back = int(args[4])
    nos_por_camada = int(args[5])
    imagepath = args[6]
    gtpath = args[7]


    fit_geral = 0
    for i in range(iter):
        print(f"Iteração {i + 1}/{iter}")
        f, p, h, hl = avaliar_imagem(imagepath, gtpath, geracoes, arglambda, nos, levels_back, nos_por_camada)
        fit_geral += f
    print(f"Fitness médio após {iter} iterações: {fit_geral / iter}")
"""som += f
#SALVAR O GENOMA DO PAI E DOS FILHOS A CADA 10 GERAÇÕES
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
grafico(h, hl, 'PR08_dibco2013')
"""
if __name__ == "__main__":
    main()