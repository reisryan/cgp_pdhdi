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

    hist_pai = []
    hist_lambda = []
    hist_stats = []

    for geracao in range(1, n_geracoes + 1):
        filhos = []
        melhores = []

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

        fitness_pai = float(pai.fitness.values[0])
        fitness_filhos = [
            float(f.fitness.values[0]) for f in filhos
        ]

        hist_pai.append(fitness_pai)
        hist_lambda.append(fitness_filhos)

        hist_stats.append({
            chave: (
                valor.item() if isinstance(valor, np.generic) else valor
            )
            for chave, valor in evolve.stats.items()
        })

        print(
            f"Geração {geracao}: pai={fitness_pai}, "
            f"filhos={fitness_filhos}"
        )
        print(f"Estatísticas: {hist_stats[-1]}")
        print(
            f"Média fitness dos filhos = "
            f"{sum(fitness_filhos) / len(fitness_filhos)}"
        )

    genoma_melhor_pai = [
        list(gene) if isinstance(gene, (list, tuple)) else gene
        for gene in pai
    ]

    cgp.genoma = pai
    executor.cgp.genoma = pai
    cgp.exibir_grafo(cgp, cgp.genoma, f"{imagepath}_melhor_grafo.png")
    nos_ativos = sorted(
        no for no in executor.nos_ativos()
        if cgp.n_entradas <= no < cgp.n_entradas + cgp.n_nos
    )

    funcoes_nos_ativos = [
        {
            "no": no,
            "operador_id": pai[no][0],
            "funcao": operadores[pai[no][0]]["funcao"].__name__
        }
        for no in nos_ativos
    ]

    return (
        float(pai.fitness.values[0]),
        pai,
        hist_pai,
        hist_lambda,
        hist_stats,
        genoma_melhor_pai,
        nos_ativos,
        funcoes_nos_ativos
    )

def main():
    if len(sys.argv) < 9:
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
                    pares_imagens.append((
                        os.path.join(pasta_original, nome),
                        caminho_gt,
                        identificador
                    ))

        if not pares_imagens:
            print("Nenhum par de imagens correspondente foi encontrado.")
            sys.exit(1)

    elif imagepath.lower() == "all" or gtpath.lower() == "all":
        print("Para executar todas as imagens, informe all all.")
        sys.exit(1)

    else:
        identificador = os.path.basename(imagepath).split("_dibco2013")[0]
        pares_imagens.append((
            os.path.join("OriginalImages", imagepath),
            os.path.join("GTimages", gtpath),
            identificador
        ))

    for caminho_imagem, caminho_gt, identificador in pares_imagens:
        if not os.path.isfile(caminho_imagem):
            print(f"Imagem não encontrada: {caminho_imagem}")
            continue

        if not os.path.isfile(caminho_gt):
            print(f"Ground truth não encontrado: {caminho_gt}")
            continue

        arquivo_resultado = f"{identificador}_result.txt"
        fitness_iteracoes = []

        print(f"\nImagem: {identificador}")

        with open(arquivo_resultado, "w", encoding="utf-8") as arquivo:
            for i in range(iteracoes):
                print(f"Iteração {i + 1}/{iteracoes}")

                (
                    f, p, hp, hl, hs,
                    genoma, nos_ativos, funcoes_ativas
                ) = avaliar_imagem(
                    caminho_imagem,
                    caminho_gt,
                    geracoes,
                    arglambda,
                    nos,
                    levels_back,
                    nos_por_camada
                )

                fitness_iteracoes.append(float(f))

                arquivo.write(f"iteracao: {i + 1}\n")
                arquivo.write(f"geracoes: {geracoes}\n")
                arquivo.write(f"lambda: {arglambda}\n")
                arquivo.write(f"nos: {nos}\n")
                arquivo.write(f"fitness_final: {float(f)}\n")
                arquivo.write(f"historico_pais: {hp!r}\n")
                arquivo.write(f"historico_lambda: {hl!r}\n")
                arquivo.write(f"historico_stats: {hs!r}\n")
                arquivo.write(f"genoma_melhor_pai: {genoma!r}\n")
                arquivo.write(f"nos_ativos_melhor_pai: {nos_ativos!r}\n")
                arquivo.write(f"quantidade_nos_ativos: {len(nos_ativos)}\n")
                arquivo.write(f"funcoes_nos_ativos: {funcoes_ativas!r}\n")
                arquivo.write("\n")
                arquivo.flush()

                print(f"Genoma final salvo para a iteração {i + 1}.")
                print(f"Nós ativos: {nos_ativos}")
                print(f"Funções ativas: {funcoes_ativas}")

            fitness_medio = sum(fitness_iteracoes) / len(fitness_iteracoes)

            arquivo.write(f"fitness_medio_iteracoes: {fitness_medio!r}\n")
            arquivo.write(f"fitness_por_iteracao: {fitness_iteracoes!r}\n")
            arquivo.write("\n")

        print(
            f"Fitness médio de {identificador} após {iteracoes} iterações: "
            f"{fitness_medio}"
        )
        print(f"Resultados salvos em {arquivo_resultado}")


if __name__ == "__main__":
    main()