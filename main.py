from deap import base, creator, tools

import sys
import re
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

def listar_pares(imagepath, gtpath):
    imgs, gts, ids = [], [], []
 
    if imagepath.lower() == "all" and gtpath.lower() == "all":
        for nome in sorted(os.listdir("OriginalImages")):
            m = re.fullmatch(r"((?:HW|PR)\d{2})_dibco2013\.(?:bmp|tif|tiff)", nome, re.IGNORECASE)
            if not m:
                continue
            ident = m.group(1)
            caminho_gt = os.path.join("GTimages", f"{ident}_estGT_dibco2013.tiff")
            if os.path.isfile(caminho_gt):
                imgs.append(os.path.join("OriginalImages", nome))
                gts.append(caminho_gt)
                ids.append(ident)
    elif imagepath.lower() == "all" or gtpath.lower() == "all":
        print("Para executar todas as imagens, informe all all.")
        sys.exit(1)
    else:
        imgs.append(os.path.join("OriginalImages", imagepath))
        gts.append(os.path.join("GTimages", gtpath))
        ids.append(os.path.basename(imagepath).split("_dibco2013")[0])
 
    return imgs, gts, ids

def avaliar_imagem(imgs, gts, n_geracoes, n_lambda, nos, levels_back, nos_por_camada, tag):
    cgp = CGP(1, operadores, nos, 1, levels_back, nos_por_camada)
    executor = CGPExecutor(cgp)
    evolve = CGPEvolve(cgp, executor, imgs, gts)
 
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individuo"):
        creator.create("Individuo", list, fitness=creator.FitnessMax)
 
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individuo, cgp.gera_genoma)
    toolbox.register("mutate", evolve.mutacao)
    toolbox.register("evaluate", evolve.fitness_medio) 
 
    pai = toolbox.individual()
    pai.fitness.values = toolbox.evaluate(pai)
 
    hist_pai, hist_lambda, hist_stats = [], [], []
 
    for geracao in range(1, n_geracoes + 1):
        filhos, melhores = [], []
 
        for _ in range(n_lambda):
            filho, = toolbox.mutate(pai)
            filho.fitness.values = toolbox.evaluate(filho)
            filhos.append(filho)
            if filho.fitness.values[0] > pai.fitness.values[0]:
                melhores.append(filho)
 
        neutros = [f for f in filhos if f.fitness.values[0] == pai.fitness.values[0]]
 
        if melhores:
            pai = random.choice(melhores)
        elif neutros:
            pai = random.choice(neutros)
 
        fitness_filhos = [float(f.fitness.values[0]) for f in filhos]
        hist_pai.append(float(pai.fitness.values[0]))
        hist_lambda.append(fitness_filhos)
        hist_stats.append({
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in evolve.stats.items()
        })
 
        print(f"Geração {geracao}: pai={hist_pai[-1]}, média filhos={sum(fitness_filhos)/len(fitness_filhos)}")
 
    psnr_por_imagem = evolve.avaliar(pai)
 
    cgp.genoma = pai
    cgp.exibir_grafo(cgp, cgp.genoma, f"{tag}_melhor_grafo.png")
    nos_ativos = sorted(
        no for no in executor.nos_ativos()
        if cgp.n_entradas <= no < cgp.n_entradas + cgp.n_nos
    )
    funcoes_ativas = [
        {"no": no, "operador_id": pai[no][0], "funcao": operadores[pai[no][0]]["funcao"].__name__}
        for no in nos_ativos
    ]
    genoma = [list(g) if isinstance(g, (list, tuple)) else g for g in pai]
 
    return float(pai.fitness.values[0]), psnr_por_imagem, genoma, nos_ativos, funcoes_ativas, hist_pai, hist_lambda, hist_stats

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
    imgs, gts, ids = listar_pares(args[6], args[7])

    if not imgs:
        print("Nenhum par de imagens correspondente foi encontrado.")
        sys.exit(1)
 
    print(f"{len(imgs)} imagens na pipeline única: {ids}")
    arquivo_resultado = "pipeline_unica_result.txt"
    fitness_iteracoes = []
 
    for i in range(iteracoes):
        print(f"\nIteração {i + 1}/{iteracoes}")
        (f, psnrs, genoma, nos_ativos, funcoes, hp, hl, hs) = avaliar_imagem(
            imgs, gts, geracoes, arglambda, nos, levels_back, nos_por_camada,
            tag=f"pipeline_unica_it{i + 1}"
        )
        fitness_iteracoes.append(f)
 
        with open(arquivo_resultado, "a", encoding="utf-8") as arq:
            arq.write(f"iteracao: {i + 1}\n")
            arq.write(f"geracoes: {geracoes}\nlambda: {arglambda}\nnos: {nos}\n")
            arq.write(f"imagens: {ids!r}\n")
            arq.write(f"fitness_final (media PSNR): {f}\n")
            arq.write(f"psnr_por_imagem: {dict(zip(ids, psnrs))!r}\n")
            arq.write(f"historico_pais: {hp!r}\n")
            arq.write(f"historico_lambda: {hl!r}\n")
            arq.write(f"historico_stats: {hs!r}\n")
            arq.write(f"genoma_melhor_pai: {genoma!r}\n")
            arq.write(f"nos_ativos_melhor_pai: {nos_ativos!r}\n")
            arq.write(f"funcoes_nos_ativos: {funcoes!r}\n\n")
 
        print(f"Fitness (média PSNR): {f}")
        print(f"PSNR por imagem: {dict(zip(ids, psnrs))}")
        print(f"Funções ativas: {funcoes}")
 
    with open(arquivo_resultado, "a", encoding="utf-8") as arq:
        arq.write(f"fitness_medio_iteracoes: {sum(fitness_iteracoes)/len(fitness_iteracoes)!r}\n")
        arq.write(f"fitness_por_iteracao: {fitness_iteracoes!r}\n\n")
 
 
if __name__ == "__main__":
    main()
 