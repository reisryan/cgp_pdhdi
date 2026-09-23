import cv2 as cv
import networkx as nx


class CGPExecutor:

    def __init__(self, cgp):
        self.cgp = cgp

    def nos_ativos(self, individuo):
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

            gene = individuo[node_id - 1]
            conexao = gene[1]
            to_verify.append(conexao)

        return nodes_enabled

    def executar(self, arquivo, individuo, grafo):
        OUTPUTS = {}

        OUTPUTS[0] = cv.imread(arquivo, cv.IMREAD_GRAYSCALE).astype("uint8")
        nodes_ativos = self.nos_ativos(individuo)

        for no in nx.topological_sort(grafo):
            if no == 0:
                continue
            if no not in nodes_ativos:
                continue

            gene = individuo[no - 1]
            operador_id = gene[0]

            funcao = self.cgp.funcoes[operador_id]["funcao"]
            conexao = gene[1]
            entrada = OUTPUTS[conexao]
            OUTPUTS[no] = funcao(entrada)

        return OUTPUTS