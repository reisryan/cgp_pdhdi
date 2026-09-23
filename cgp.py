import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# TIPAGEM DE DADOS

class Imagem:
    pass


class ImagemBinaria:
    pass


class CGP:
    def __init__(self, n_entradas, funcoes, n_nos, n_saidas):
        self.n_entradas = n_entradas
        self.funcoes = funcoes
        self.n_nos = n_nos
        self.n_saidas = n_saidas
        self.genoma = None


    def construcao_grafo(self, individuo):
        grafo = nx.DiGraph()

        # Nó de entrada
        grafo.add_node(0, tipo=Imagem)

        for i, gene in enumerate(individuo[:-1], start=1):

            operador_id = gene[0]
            operador = self.funcoes[operador_id]

            grafo.add_node(
                i,
                operador=operador_id,
                entradas=operador["entrada"],
                saida=operador["saida"],
                tipo=operador["saida"]
            )

            grafo.add_edge(gene[1], i)

        return grafo

    def verificacao_tipos(self, individuo, grafo):
        for i, gene in enumerate(individuo, start=1):

            operador_id = gene[0]
            conexao = gene[1]

            operador = self.funcoes[operador_id]

            tipo_origem = grafo.nodes[conexao]["tipo"]
            tipo_esperado = operador["entrada"]

            if tipo_origem != tipo_esperado:
                return False

        return True

    def conexoes_validas(self, grafo, operador, no):
        conexoes = []

        for i in grafo.nodes:

            # Só pode conectar a nós anteriores
            if i >= no:
                continue

            if grafo.nodes[i]["tipo"] == self.funcoes[operador]["entrada"]:
                conexoes.append(i)

        return conexoes

    def gera_individuo(self):

        individuo = []

        grafo = nx.DiGraph()
        grafo.add_node(0, tipo=Imagem)

        for no in range(1, self.n_nos + 1):

            operadores_validos = []

            for operador in self.funcoes:

                conexoes = self.conexoes_validas(
                    grafo,
                    operador,
                    no
                )

                if conexoes:
                    operadores_validos.append(operador)

            operador = np.random.choice(operadores_validos)

            conexoes = self.conexoes_validas(
                grafo,
                operador,
                no
            )

            conexao = np.random.choice(conexoes)

            individuo.append([
                operador,
                conexao
            ])

            grafo.add_node(
                no,
                operador=operador,
                tipo=self.funcoes[operador]["saida"]
            )

            grafo.add_edge(conexao, no)

        # Saídas
        for _ in range(self.n_saidas):

            saida = np.random.randint(
                1,
                self.n_nos + 1
            )

            individuo.append([
                "out",
                saida
            ])

        return individuo

    def exibir_grafo(self, individuo, salvar=None):
        grafo = self.construcao_grafo(individuo)

        pos = nx.spring_layout(
            grafo,
            seed=1
        )

        labels = {}

        for no in grafo.nodes:

            if no == 0:
                labels[no] = "INPUT"

            else:
                operador_id = grafo.nodes[no]["operador"]
                nome = self.funcoes[operador_id]["funcao"].__name__
                labels[no] = f"{no}: {nome}"

        nx.draw(
            grafo,
            pos,
            labels=labels,
            with_labels=True,
            node_size=2000,
            arrows=True,
            font_size=9
        )

        if salvar:
            plt.savefig(
                salvar,
                dpi=150,
                bbox_inches="tight"
            )

        plt.show()