# IMPORTAÇÕES #

from deap import base, creator, tools, gp, algorithms
from skimage.filters import threshold_multiotsu, threshold_sauvola

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import cv2 as cv
import numpy as np
import os

cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_ERROR)
np.random.seed(1)

# TIPAGEM DE DADOS #

class Imagem:
    pass
class ImagemBinaria:
    pass

#Estrutura do INDIVIDUO (genoma)
 # nó 1: [operador, nó (operação anterior)]
 # nó n(lasts) : [out, nó de saída]
 # OUTPUTS[0] = imagem inicial
 # OUTPUTS[1::n] = saída de cada nó

# FUNÇÕES GERAIS E DE PROCESSAMENTO DE IMAGEM #

def construcao_grafo(individuo):
    grafo = nx.DiGraph()
    grafo.add_node(0,tipo=Imagem)

    for i, gene in enumerate(individuo[:-1], start=1):
        operador = operadores[gene[0]]
        grafo.add_node(i, operador=gene[0], entradas=operador["entrada"], saida=operador["saida"], tipo=operador["saida"])
        grafo.add_edge(gene[1], i)
    return grafo

def verificacao_tipos(individuo, grafo):
    for i, gene in enumerate(individuo, start=1):
        nome_op=gene[0]
        conexao=gene[1]

        operador = operadores[nome_op]
        tipo_origem = grafo.nodes[conexao]["tipo"]
        tipo_esperado = operador["entrada"][0]

        if tipo_origem != tipo_esperado:
            return False
    return True

def execucao_individuo(file, individuo, grafo):
    OUTPUTS = {}
    OUTPUTS[0] = cv.imread(f'{file}', cv.IMREAD_GRAYSCALE).astype(np.uint8)
    
    for no in nx.topological_sort(grafo):
        if no == 0:
            continue
        if no in nos_ativos(individuo):
            gene = individuo[no-1]
            operador = gene[0]
            funcao = operadores[operador]["funcao"]
            conexao = gene[1]
            entrada = OUTPUTS[conexao]
            OUTPUTS[no] = funcao(entrada)        
    return OUTPUTS

def nos_ativos(individuo):
    nodes_enabled = set()

    to_verify = []
    to_verify.append(individuo[-1][1])

    while to_verify:
        node_id = to_verify.pop()

        if node_id in nodes_enabled:
            continue

        nodes_enabled.add(node_id)

        if node_id == 0:
            continue

        gene = individuo[node_id-1]
        conexao = gene[1]

        to_verify.append(conexao)

    return nodes_enabled

def erro(TARGET, resultado):
    err = np.mean((TARGET.astype(np.float64) - resultado.astype(np.float64))**2) #MSE
    return err

def fitness_individuo(OUTPUTS, TARGET, individuo):
    fitness = 0

    saida = individuo[-1][1]
    fitness = erro(TARGET, OUTPUTS[saida])
    return fitness

def conexoes_validas(grafo, operador, no):
    conexoes = []
    for i in grafo.nodes:
        if i >= no:
            continue
        if grafo.nodes[i]["tipo"] == operadores[operador]["entrada"][0]:
            conexoes.append(i)
        
    return conexoes

def gera_individuo(N_NOS=8, N_SAIDAS=1):
    individuo = []
    
    grafo = nx.DiGraph()
    grafo.add_node(0, tipo=Imagem)

    for no in range(1, N_NOS+1):
        op_validos = []
        for op in operadores:
            conexoes = conexoes_validas(grafo, op, no)
            if conexoes:
                op_validos.append(op)

        operador = np.random.choice(op_validos)
        conexoes = conexoes_validas(grafo, operador, no)
        conexao = np.random.choice(conexoes)

        individuo.append([operador, conexao])

        grafo.add_node(no, operador=operador, tipo=operadores[operador]["saida"])
        grafo.add_edge(conexao, no)

    for i in range(N_SAIDAS):
        saida = np.random.randint(1, N_NOS + 1)
        individuo.append(["out", saida])
    return individuo

def avaliar(individuo, input, output):

    grafo = construcao_grafo(individuo)

    if not verificacao_tipos(individuo[:-1], grafo):
        return (float("inf"),)
    erros = []

    for image_input_path, image_output_path in zip(input, output):
        img_output = cv.imread(f'{image_output_path}', cv.IMREAD_GRAYSCALE).astype(np.uint8)
        OUTPUTS = execucao_individuo(image_input_path, individuo, grafo)
        erro_img = fitness_individuo(OUTPUTS, img_output, individuo)
        erros.append(erro_img)
    fitness = np.mean(erros)

    return (fitness,)

def mutacao(individuo):

    while True:
        mutation_rate = 0.5

        filho = creator.Individuo(individuo)
        indice = np.random.randint(0, len(filho)-1)
        no = indice + 1

        grafo = construcao_grafo(filho)

        operadores_validos = []

        for nome in operadores:
            conexoes = conexoes_validas(grafo, nome, no)

            if conexoes:
                operadores_validos.append(nome)

        if np.random.random() < mutation_rate:
            operador = np.random.choice(operadores_validos)
            conexoes = conexoes_validas(grafo, operador, no)
            conexao = np.random.choice(conexoes)
            filho[indice] = [operador, conexao]
        else:
            saida = np.random.randint(1, len(filho)-1)
            filho[-1] = ["out", saida]
        
        grafo_novo = construcao_grafo(filho)

        if verificacao_tipos(filho[:-1], grafo_novo):
            return filho,

def clahe(imagem):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(imagem)

def limiares(imagem):
    lim = threshold_multiotsu(imagem, classes=3)
    return lim

def otsu(imagem):
    lim = limiares(imagem)
    _, imgbin = cv.threshold(imagem, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return imgbin

def multiotsu(imagem):
    lim = limiares(imagem)
    _, imgbin = cv.threshold(imagem, lim[0], 255, cv.THRESH_BINARY)
    return imgbin

def nick(imagem):
    ws = 7
    k_nick = -0.05  # Parâmetro de controle da influência do desvio padrão local.
    img_f = imagem.astype(np.float64)
    
    # Cálculo eficiente da média local m(x, y) usando Box Filter (filtro de caixa 2D uniforme).
    m_img = cv.boxFilter(img_f, -1, (ws, ws), normalize=True)
    
    # Cálculo da média dos quadrados locais para computar a variância local de forma otimizada.
    img_sq = img_f ** 2
    sum_sq_img = cv.boxFilter(img_sq, -1, (ws, ws), normalize=False)
    
    NP = ws * ws
        # Cálculo do radicando (estimador de variância/desvio padrão da vizinhança).
        # Garante-se que valores levemente negativos causados por precisão numérica sejam zerados.
    radicand = (sum_sq_img - (m_img ** 2)) / NP
    radicand = np.maximum(radicand, 0.0)
    
        # Geração do mapa bidimensional de limiares locais T_nick_img.
    T_nick_img = m_img + k_nick * np.sqrt(radicand)
    
        # Segmentação local: pixels com intensidade menor ou igual ao limiar local viram preto (0),
        # e pixels maiores viram branco (255).
    img_nick = np.zeros_like(imagem, dtype=np.uint8)
    img_nick[imagem <= T_nick_img] = 0
    img_nick[imagem > T_nick_img] = 255
    
        # ==============================================================================
        # 5. DETECÇÃO DE MANCHAS (SMEAR) POR DENSIDADE E MESCLAGEM HÍBRIDA
        # ==============================================================================
        # Manchas ou vazamento de tinta (smear) degradam a binarização global.
        # Este bloco divide a imagem binarizada global em blocos não-sobrepostos,
        # calcula a densidade de pixels pretos (texto/mancha) em cada bloco e
        # rotula blocos com densidade anormalmente alta (outliers) como manchas.
    
    k_smear = 0.001
    h, w = imagem.shape
    fSlist = []
    blockscoords = []
    
        # Partição da imagem em blocos disjuntos de dimensão ws x ws.
    for y in range(0, h, ws):
        for x in range(0, w, ws):
            block = imagem[y:y+ws, x:x+ws]
            tp = block.size
            if tp > 0:
                # Quantidade de pixels pretos no bloco.
                bp = np.sum(block == 0)
                fS = bp / tp                 # Fração de pixels pretos (densidade).
                fSlist.append(fS)
                blockscoords.append((y, x, fS, block.shape))
    
        # Média e desvio padrão global da densidade de pixels pretos por bloco.
    m_fS, s_fS = np.mean(fSlist), np.std(fSlist)
    
    smear_mask = np.zeros_like(imagem)
    
        # Classificação dos blocos: se a fração de pretos exceder (média + k_smear * desvio_padrão),
        # a região é classificada como mancha ou região densa de interesse.
    for y, x, fS, shape in blockscoords:
        if fS > (m_fS + k_smear * s_fS):
            smear_mask[y:y+shape[0], x:x+shape[1]] = 255
    
        # Dilatação morfológica da máscara de manchas para abranger as bordas e áreas de transição.
    kernel_dilate_mask = np.ones((ws, ws), np.uint8)
    smear_mask = cv.dilate(smear_mask, kernel_dilate_mask, iterations=1)
    
        # Mesclagem híbrida (Hybrid Binarization):
        # - Nas áreas cobertas pela máscara de mancha (smear_mask == 255), utiliza-se o resultado
        #   do algoritmo local adaptativo (img_nick), que é mais robusto a variações locais de iluminação.
        # - Nas áreas limpas de fundo (fundo normal), preserva-se o resultado da binarização global.
    img_b2 = np.where(smear_mask == 255, img_nick, imagem)
    return img_b2

def sauvola(imagem):
    ws = 17
    thresh_sauvola = threshold_sauvola(imagem, window_size=ws, k=0.2)
    img_sauvola = (imagem > thresh_sauvola).astype("uint8") * 255
    return img_sauvola

def open_morphology(imagem):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    return cv.morphologyEx(imagem, cv.MORPH_OPEN, kernel)

def close_morphology(imagem):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    return cv.morphologyEx(imagem, cv.MORPH_CLOSE, kernel)

# PRIMITIVAS ou FUNÇÕES #

operadores = {
    0: {"funcao": clahe, "entrada": [Imagem], "saida": Imagem},
    1: {"funcao": otsu, "entrada": [Imagem], "saida": ImagemBinaria},
    2: {"funcao": multiotsu, "entrada": [Imagem], "saida": ImagemBinaria},
    3: {"funcao": nick, "entrada": [Imagem], "saida": ImagemBinaria},
    4: {"funcao": nick, "entrada": [ImagemBinaria], "saida": ImagemBinaria},
    5: {"funcao": sauvola, "entrada": [Imagem], "saida": ImagemBinaria},
    6: {"funcao": sauvola, "entrada": [ImagemBinaria], "saida": ImagemBinaria},
    7: {"funcao": open_morphology, "entrada": [ImagemBinaria], "saida": ImagemBinaria},
    8: {"funcao": close_morphology, "entrada": [ImagemBinaria], "saida": ImagemBinaria}
}


# FITNESS - FUNÇÃO DE AVALIAÇÃO #

# Mminimizar o resultado de erro entre a imagem resultante e a imagem esperada
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individuo, gera_individuo)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)



input, output = [], []
pasta = 'OriginalImages'
for image in os.listdir(pasta):
    nome = os.path.splitext(image)[0]
    input.append(f'{pasta}/{image}')
    output.append(f'GTimages/{nome}_estGT.tiff')
    
toolbox.register("evaluate", avaliar, input=input, output=output)
    

populacao = toolbox.population(n=2)

toolbox.register("mutate", mutacao)

fitness_history = []

for individuo in populacao:
    individuo.fitness.values = toolbox.evaluate(individuo)

pai = tools.selBest(populacao, 1)[0]
neutros_geracao = []
for geracao in range(50):
    
    filhos = []

    for i in range(8):

        filho, = toolbox.mutate(pai)

        filho.fitness.values = toolbox.evaluate(filho)

        filhos.append(filho)

    melhor_filho = tools.selBest(filhos, 1)[0]    
    neutros = sum(1 for f in filhos if f.fitness.values[0] == pai.fitness.values[0])
    print(f"Geração {geracao}: {neutros} indivíduos neutros")
    neutros_geracao.append(neutros)
    if melhor_filho.fitness.values[0] <= pai.fitness.values[0]:
        pai = melhor_filho
        print(f"Geração {geracao}: "
        f"fitness = {pai.fitness.values[0]}")
    fitness_history.append(pai.fitness.values[0])


plt.plot(fitness_history)
plt.yscale('log')
plt.xlabel('Geração')
plt.ylabel('Erro (MSE)')
plt.title('Evolução do Fitness')
plt.savefig("fitness.png", dpi=150)

# MELHOR INDIVÍDUO # 

melhor = pai

print("MELHOR INDIVÍDUO:")
print(melhor)

print("FITNESS:")
print(melhor.fitness.values)
