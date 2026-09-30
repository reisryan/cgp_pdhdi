import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import functions_pi as fpi
from cgpExecutor import CGPExecutor

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
        self.tam_genoma = self.n_nos * 2 + self.n_saidas

    def construcao_grafo(self, individuo):
        grafo = nx.DiGraph()
        for i in range(self.n_entradas):
            grafo.add_node(i, operador="input", tipo=Imagem, categoria="input")

        for no in range(self.n_entradas, self.n_entradas + self.n_nos):
            operador_id, conexao = individuo[no]

            grafo.add_node(no, operador=operador_id,  tipo=self.funcoes[operador_id]["saida"], categoria="node_function")
            grafo.add_edge(conexao, no)

        for j in range(self.n_entradas+self.n_nos, self.n_entradas+self.n_nos+self.n_saidas):
            _, conexao = individuo[j]
            
            grafo.add_node(j, operador="output", tipo=self.funcoes[individuo[conexao][0]]["saida"], categoria="output")
            grafo.add_edge(conexao, j)

        return grafo

    def verificacao_tipos(self, individuo):
        for no in range(self.n_entradas, self.n_entradas + self.n_nos):
            operador_id, conexao = individuo[no]
            operador = self.funcoes[operador_id]

            if conexao < self.n_entradas:
                tipo_origem = Imagem
            else:
                operador_origem = individuo[conexao][0]
                tipo_origem = self.funcoes[operador_origem]["saida"]
                
            tipo_esperado = operador["entrada"]

            if tipo_origem != tipo_esperado:
                return False

        return True

    def conexoes_validas(self, individuo, operador, no):
        conexoes = []

        for i in range(no):
            if i < self.n_entradas:
                tipo_saida = Imagem
            else:
                operador_anterior = individuo[i][0]
                tipo_saida = self.funcoes[operador_anterior]["saida"]

            if tipo_saida == self.funcoes[operador]["entrada"]:
                conexoes.append(i)

        return conexoes

    def gera_genoma(self):
        individuo = []
    # LEVELS BACK -> ACRESCENTAR
        for i in range(0,self.n_entradas):
            individuo.append([i, i])
        
        for no in range(self.n_entradas, self.n_entradas + self.n_nos):
            operadores_validos = []

            for operador in self.funcoes:
                conexoes = self.conexoes_validas(individuo, operador, no)
                if conexoes:
                    operadores_validos.append(operador)

            operador = np.random.choice(operadores_validos)
            conexoes = self.conexoes_validas(individuo,operador, no)
            conexao = np.random.choice(conexoes)
            individuo.append([operador,conexao])

        for _ in range(self.n_saidas):
            saida = np.random.randint(self.n_entradas, self.n_entradas + self.n_nos)
            individuo.append(["out", saida])

        self.genoma = individuo
        return self.genoma

    def exibir_grafo(self, cgp, individuo, salvar=None):
        grafo = self.construcao_grafo(individuo)

        # Descobre quais nós estão ativos
        executor = CGPExecutor(cgp)
        cgp.genoma = individuo
        nos_ativos = executor.nos_ativos()

        pos = {}
        labels = {}
        cores = {}

        # -------------------------
        # POSIÇÕES
        # -------------------------

        # Entradas
        for i in range(self.n_entradas):
            pos[i] = (0, 0)

        # Nós de função
        for j in range(self.n_entradas, self.n_entradas + self.n_nos):
            if j % 2 == 0:
                pos[j] = (1, -(j- self.n_entradas))
            else:
                pos[j] = (2, -(j- self.n_entradas))

        # Saídas
        for k in range(
            self.n_entradas + self.n_nos,
            self.n_entradas + self.n_nos + self.n_saidas
        ):
            pos[k] = (3, 0)

        # -------------------------
        # LABELS E CORES
        # -------------------------

        for no in grafo.nodes:

            categoria = grafo.nodes[no]["categoria"]

            if categoria == "input":
                labels[no] = f"I-N{no}"
                cores[no] = "lightblue"

            elif categoria == "node_function":

                operador_id = grafo.nodes[no]["operador"]
                nome = self.funcoes[operador_id]["funcao"].__name__

                labels[no] = f"F-N{no}\n({nome})"

                if no in nos_ativos:
                    cores[no] = "red"
                else:
                    cores[no] = "green"

            else:
                labels[no] = f"O-N{no}"
                cores[no] = "orange"

        # -------------------------
        # DESENHO
        # -------------------------

        plt.figure(figsize=(14, 10))

        nx.draw(
            grafo,
            pos,
            labels=labels,
            with_labels=True,
            node_color=[cores[no] for no in grafo.nodes],
            node_size=2200,
            arrows=True,
            arrowsize=15,
            font_size=8,
            edge_color="black"
        )

        if salvar:
            plt.savefig(
                salvar,
                dpi=150,
                bbox_inches="tight"
            )

        plt.show()
        plt.close()