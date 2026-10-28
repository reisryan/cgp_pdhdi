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
        hist_pai.append(float(pai.fitness.values[0]))
        hist_lambda.append([float(f.fitness.values[0]) for f in filhos])

        if geracao % 10 == 0:
            hist_geracao.append((geracao, hist_pai, hist_lambda))
            
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
    h_g = []
    for i in range(iter):
        print(f"Iteração {i + 1}/{iter}")
        f, p, hp, hl = avaliar_imagem('OriginalImages/' + imagepath, 'GTimages/' + gtpath, geracoes, arglambda, nos, levels_back, nos_por_camada)
        file = open(f"{imagepath}_result.txt", "a")
        file.write(f"Pai: {hp}\n")
        file.write(f"Filhos: {hl}\n")
        file.write("\n")
        file.close()
        fit_geral += f
    print(f"Fitness médio após {iter} iterações: {fit_geral / iter}")
    
    
if __name__ == "__main__":
    main()