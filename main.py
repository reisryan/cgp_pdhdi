from deap import base, creator, tools

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import functions_pi as fpi
from cgp import CGP, Imagem, ImagemBinaria
from cgpExecutor import CGPExecutor
from cgpEvolve import CGPEvolve


np.random.seed(1)



operadores = {
    0: {"funcao": fpi.clahe, "entrada": Imagem, "saida": Imagem},
    1: {"funcao": fpi.otsu, "entrada": Imagem, "saida": ImagemBinaria},
    2: {"funcao": fpi.multiotsu, "entrada": Imagem, "saida": ImagemBinaria},
    3: {"funcao": fpi.nick, "entrada": Imagem, "saida": ImagemBinaria},
    4: {"funcao": fpi.nick, "entrada": ImagemBinaria, "saida": ImagemBinaria},
    5: {"funcao": fpi.sauvola, "entrada": Imagem, "saida": ImagemBinaria},
    6: {"funcao": fpi.sauvola, "entrada": ImagemBinaria, "saida": ImagemBinaria},
    7: {"funcao": fpi.open_morphology, "entrada": ImagemBinaria,"saida": ImagemBinaria},
    8: {"funcao": fpi.close_morphology, "entrada": ImagemBinaria, "saida": ImagemBinaria},
    9: {"funcao": fpi.square_root, "entrada": Imagem, "saida": Imagem},
    10: {"funcao": fpi.square, "entrada": Imagem, "saida": Imagem},
    11: {"funcao": fpi.f_exp, "entrada": Imagem, "saida": Imagem},
    12: {"funcao": fpi.f_log, "entrada": Imagem, "saida": Imagem},
    13: {"funcao": fpi.median_blur, "entrada": Imagem, "saida": Imagem},
    14: {"funcao": fpi.gaussian_blur, "entrada": Imagem, "saida": Imagem},
    15: {"funcao": fpi.laplacian, "entrada": Imagem, "saida": Imagem},
    16: {"funcao": fpi.sobel, "entrada": Imagem, "saida": Imagem},
    17: {"funcao": fpi.erode, "entrada": ImagemBinaria, "saida": ImagemBinaria},
    18: {"funcao": fpi.dilate, "entrada": ImagemBinaria, "saida": ImagemBinaria},
    19: {"funcao": fpi.normalize, "entrada": Imagem, "saida": Imagem},
    20: {"funcao": fpi.f_invert, "entrada": ImagemBinaria, "saida": ImagemBinaria},
    21: {"funcao": fpi.bilateral_filter, "entrada": Imagem, "saida": Imagem},
    22: {"funcao": fpi.morph_gradient, "entrada": Imagem, "saida": Imagem},
    23: {"funcao": fpi.f_sharpen, "entrada": Imagem, "saida": Imagem}
}

# CGP

cgp = CGP(1, operadores, 10, 1)
executor = CGPExecutor(cgp)
evolve = CGPEvolve(cgp, executor)

# DEAP

creator.create("FitnessMin", base.Fitness,  weights=(-1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individuo, cgp.gera_genoma)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

input, output = [], []
pasta = 'OriginalImages'

for image in os.listdir(pasta):
    nome = os.path.splitext(image)[0]
    input.append(f'{pasta}/{image}')
    output.append(f'GTimages/{nome}_estGT.tiff')
    break 

toolbox.register("evaluate", evolve.avaliar, inputs=input, outputs=output)
populacao = toolbox.population(n=2)
toolbox.register("mutate", evolve.mutacao)

fitness_history = []
for individuo in populacao:
    individuo.fitness.values = toolbox.evaluate(individuo)
pai = tools.selBest(populacao, 1)[0]
neutros_geracao = []


#SALVAR A CADA 10 GERAÇÕES EM ALGUM ARQUIVO OS INDIVIDUOS, E, A PARTIR DISTO, FAZER OS GRAFOS/GRAFICOS
for geracao in range(25):
    filhos = []
    for i in range(5): 
        filho, = toolbox.mutate(pai)
        filho.fitness.values = toolbox.evaluate(filho)
        filhos.append(filho)
    melhor_filho = tools.selBest(filhos, 1)[0]
    neutros = sum(1 for f in filhos if f.fitness.values[0] == pai.fitness.values[0])
    print(f"Geração {geracao}: {neutros} indivíduos neutros")
    neutros_geracao.append(neutros)
    if melhor_filho.fitness.values[0] <= pai.fitness.values[0]:
        pai = melhor_filho
    print(f"Geração {geracao}: " f"fitness = {pai.fitness.values[0]}")
    fitness_history.append(pai.fitness.values[0])
"""
    plt.plot(fitness_history)
    plt.yscale('log')
    plt.xlabel('Geração x Indivíduos')
    plt.ylabel('Erro (MSE)')
    plt.title('Evolução do Fitness')
    plt.savefig("fitness.png", dpi=150)
"""

# MELHOR INDIVÍDUO #
melhor = pai
print("MELHOR INDIVÍDUO:")
print(melhor)
print("FITNESS:")
print(melhor.fitness.values)
cgp.exibir_grafo(cgp,
    melhor,
    salvar="melhor_grafo.png"
)