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
    global cgp

    cgp = CGP(1, operadores, nos, 1, levels_back, nos_por_camada)
    executor = CGPExecutor(cgp)
    evolve = CGPEvolve(cgp, executor)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individuo"):
        creator.create("Individuo", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individuo, cgp.gera_genoma)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", evolve.mutacao)
    toolbox.register("evaluate", evolve.avaliar, inputs=[imagepath], outputs=[outputpath])

    pai = toolbox.individual()
    pai.fitness.values = toolbox.evaluate(pai)

    hist_pai, hist_lambda, hist_stats = [], [], []

    for geracao in range(1, n_geracoes + 1):
        filhos, melhores = [], []

        for l in range(n_lambda):
            filho, = toolbox.mutate(pai)
            filho.fitness.values = toolbox.evaluate(filho)
            filhos.append(filho)

            if filho.fitness.values[0] > pai.fitness.values[0]:
                melhores.append(filho)

        neutros = [
            f for f in filhos
            if f.fitness.values[0] == pai.fitness.values[0]
        ]

        if melhores:
            pai = random.choice(melhores)
        elif neutros:
            pai = random.choice(neutros)

        print(
            f"Geração {geracao}: FITNESS, pai: {pai.fitness.values[0]}, "
            f"filhos: {[float(f.fitness.values[0]) for f in filhos]}"
        )
        print(
            f"nos ativos: {evolve.stats['nos_ativos']}, "
            f"mutações válidas: {evolve.stats['validas']}, "
            f"mutações descartadas: {evolve.stats['descartadas']} \n"
            f"mutações em nós: {evolve.stats['mut_no']}, "
            f"mutações em arestas: {evolve.stats['mut_aresta']}, "
            f"mutações em saídas: {evolve.stats['mut_saida']}, "
            f"mutações em nós ativos: {evolve.stats['mut_ativa']}, "
            f"mutações em nós inativos: {evolve.stats['mut_inativa']}"
        )
        print(
            f"Média fitness = "
            f"{sum(f.fitness.values[0] for f in filhos) / len(filhos)}"
        )

        if geracao % 10 == 0:
            hist_pai.append(float(pai.fitness.values[0]))
            hist_lambda.append(
                [float(f.fitness.values[0]) for f in filhos]
            )
            hist_stats.append(dict(evolve.stats))

    return pai.fitness.values[0], pai, hist_pai, hist_lambda, hist_stats


def main():
    if len(sys.argv) < 8:
        print(
            "Uso: python main.py <iteracoes> <geracoes> <lambda> "
            "<nos> <levels_back> <nos_por_camada> <imagepath> <gtpath>"
        )
        sys.exit(1)

    args = sys.argv[1:]
    iteracoes = int(args[0])
    geracoes = int(args[1])
    arglambda = int(args[2])
    nos = int(args[3])
    levels_back = int(args[4])
    nos_por_camada = int(args[5])
    imagepath = args[6]
    gtpath = args[7]

    pares_imagens = []

    if imagepath.lower() == "all" and gtpath.lower() == "all":
        import re

        pasta_original = "OriginalImages"
        pasta_gt = "GTimages"

        for nome in sorted(os.listdir(pasta_original)):
            correspondencia = re.fullmatch(
                r"((?:HW|PR)\d{2})_dibco2013\.(?:bmp|tif|tiff)",
                nome,
                re.IGNORECASE
            )

            if correspondencia:
                identificador = correspondencia.group(1)
                nome_gt = f"{identificador}_estGT_dibco2013.tiff"
                caminho_gt = os.path.join(pasta_gt, nome_gt)

                if os.path.isfile(caminho_gt):
                    pares_imagens.append(
                        (
                            os.path.join(pasta_original, nome),
                            caminho_gt,
                            identificador
                        )
                    )

        if not pares_imagens:
            print("Nenhum par de imagens correspondente foi encontrado.")
            sys.exit(1)

    elif imagepath.lower() == "all" or gtpath.lower() == "all":
        print("Para executar todas as imagens, informe all all.")
        sys.exit(1)

    else:
        identificador = os.path.basename(imagepath).split("_dibco2013")[0]
        pares_imagens.append(
            (
                os.path.join("OriginalImages", imagepath),
                os.path.join("GTimages", gtpath),
                identificador
            )
        )

    for caminho_imagem, caminho_gt, identificador in pares_imagens:
        if not os.path.isfile(caminho_imagem):
            print(f"Imagem não encontrada: {caminho_imagem}")
            continue

        if not os.path.isfile(caminho_gt):
            print(f"Ground truth não encontrado: {caminho_gt}")
            continue

        arquivo_resultado = f"{identificador}_result.txt"
        fit_geral = 0

        print(f"\nImagem: {identificador}")

        with open(arquivo_resultado, "a", encoding="utf-8") as arquivo:
            for i in range(iteracoes):
                print(f"Iteração {i + 1}/{iteracoes}")

                f, p, hp, hl, hs = avaliar_imagem(
                    caminho_imagem,
                    caminho_gt,
                    geracoes,
                    arglambda,
                    nos,
                    levels_back,
                    nos_por_camada
                )

                fit_geral += f

                for indice, stats in enumerate(hs):
                    geracao_checkpoint = (indice + 1) * 10

                    arquivo.write(f"Iteração {i + 1}: geração {geracao_checkpoint}\n")
                    arquivo.write(f"Histórico pai: {hp[:indice + 1]}\n")
                    arquivo.write(f"Histórico lambda: {hl[:indice + 1]}\n")
                    arquivo.write("evolve.stats:\n")
                    for atributo, valor in stats.items():
                        arquivo.write(f"{atributo}: {valor}\n")
                    arquivo.write("\n")
                arquivo.flush()
            arquivo.write(
                f"Fitness médio após {iteracoes} iterações: "
                f"{fit_geral / iteracoes}\n\n"
            )

        print(
            f"Fitness médio de {identificador} após {iteracoes} iterações: "
            f"{fit_geral / iteracoes}"
        )
        print(f"Resultados salvos em {arquivo_resultado}")
    
if __name__ == "__main__":
    main()