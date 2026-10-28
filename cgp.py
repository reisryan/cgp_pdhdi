import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import functions_pi as fpi
import math
from cgpExecutor import CGPExecutor

class Imagem:
    pass

class ImagemBinaria:
    pass

class CGP:
    def __init__(self, n_entradas, funcoes, n_nos, n_saidas, levels_back=1, nos_por_camada=3):
        self.n_entradas = n_entradas
        self.funcoes = funcoes
        self.n_funcoes = len(funcoes)
        self.n_nos = n_nos
        self.n_saidas = n_saidas

        self.levels_back = levels_back
        self.nos_por_camada = nos_por_camada
        self.n_camadas = math.ceil((self.n_nos) / self.nos_por_camada)

        self.genoma = None
        self.tam_genoma = self.n_nos * 2 + self.n_saidas

    def camada_do_no(self, no):
        return (no - self.n_entradas) // self.nos_por_camada

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
        if self.funcoes[operador]["entrada"] is Imagem:
            conexoes.extend(range(self.n_entradas))

        camada_atual = self.camada_do_no(no)
        primeira_camada = max(0, camada_atual - self.levels_back)
        ultima_camada = camada_atual - 1

        for i in range(self.n_entradas, no):
            camada_i = self.camada_do_no(i)
            if not (primeira_camada <= camada_i <= ultima_camada):
                continue

            operador_anterior = individuo[i][0]
            tipo_saida = self.funcoes[operador_anterior]["saida"]

            if tipo_saida == self.funcoes[operador]["entrada"]:
                conexoes.append(i)
        return conexoes

    def gera_genoma(self):
        individuo = []
        for i in range(0,self.n_entradas):
            individuo.append([i, i])
        
        for no in range(self.n_entradas, self.n_entradas + self.n_nos):
            operadores_validos = []

            for operador in self.funcoes:
                conexoes = self.conexoes_validas(individuo, operador, no)
                if conexoes:
                    operadores_validos.append(operador)
            if operadores_validos:
                operador = np.random.choice(operadores_validos)
                conexoes = self.conexoes_validas(individuo,operador, no)
                conexao = np.random.choice(conexoes)
                individuo.append([operador,conexao])

        for _ in range(self.n_saidas):
            saida = np.random.randint(self.n_entradas, self.n_entradas + self.n_nos)
            individuo.append(["out", saida])

        self.genoma = individuo
        return self.genoma

    def profundidades(self, individuo):
        prof = {i: 0 for i in range(self.n_entradas)}
        for no in range(self.n_entradas, self.n_entradas + self.n_nos):
            func, order = individuo[no]
            prof[no] = prof[order] + 1
        return prof

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

    def exibir_grafo(self, cgp, individuo, salvar=None):
        grafo = self.construcao_grafo(individuo)

        cgp.genoma = individuo
        executor = CGPExecutor(cgp)
        nos_ativos = executor.nos_ativos()

        prof = self.profundidades(individuo)
        prof_max = max(prof.values())
        
        colunas = {}
        for no in grafo.nodes:
            categoria = grafo.nodes[no]["categoria"]
            if categoria == "input":
                colunas[no] = 0
            elif categoria == "node_function":
                colunas[no] = prof[no]
            else:
                colunas[no] = prof_max + 1        

        pos = {}

        ocupacao = {}
        proxima_linha = 0

        for i in range(self.n_entradas):
            pos[i] = (0, 0)
            ocupacao.setdefault(0, set()).add(0)

        nos_funcao = list(range(self.n_entradas, self.n_entradas + self.n_nos))

        for no in nos_funcao:
            if no not in nos_ativos:
                continue
            coluna = colunas[no]
            predecessores = list(grafo.predecessors(no))
            if not predecessores:
                linha_desejada = proxima_linha
            else:
                pred = predecessores[0]
                if pred in pos:
                    linha_desejada = pos[pred][1]
                else:
                    linha_desejada = proxima_linha
            linha = linha_desejada
            ocupacao.setdefault(coluna, set())

            while linha in ocupacao[coluna]:
                linha += 1

            pos[no] = (coluna, -linha)
            ocupacao[coluna].add(linha)
            proxima_linha = max(proxima_linha, linha + 1)

        for no in grafo.nodes:
            if grafo.nodes[no]["categoria"] != "output":
                continue

            coluna = colunas[no]
            predecessores = list(grafo.predecessors(no))

            if predecessores:
                pred = predecessores[0]
                if pred in pos:
                    linha = pos[pred][1]
                else:
                    linha = 0
            else:
                linha = 0

            pos[no] = (coluna, linha)

        # NÓS INATIVOS

        linha_inativos = -(proxima_linha + 3)

        for no in nos_funcao:

            if no in nos_ativos:
                continue

            coluna = colunas[no]

            # Cada coluna recebe seus inativos em sequência
            quantidade_inativos = sum(
                2.6
                for n in nos_funcao
                if n not in nos_ativos
                and colunas[n] == coluna
                and n < no
            )

            pos[no] = (
                coluna,
                linha_inativos - quantidade_inativos
            )

        # --------------------------------------------------
        # LABELS E CORES
        # --------------------------------------------------

        labels = {}
        cores = {}

        for no in grafo.nodes:

            categoria = grafo.nodes[no]["categoria"]

            if categoria == "input":

                labels[no] = f"N{no}"
                cores[no] = "lightblue"

            elif categoria == "node_function":

                operador_id = grafo.nodes[no]["operador"]

                nome = self.funcoes[
                    operador_id
                ]["funcao"].__name__

                labels[no] = f"N{no}\n({nome})"

                if no in nos_ativos:
                    cores[no] = "red"
                else:
                    cores[no] = "green"

            else:

                labels[no] = f"N{no}"
                cores[no] = "orange"

        # --------------------------------------------------
        # SEPARA ARESTAS ATIVAS E INATIVAS
        # --------------------------------------------------

        arestas_ativas = []
        arestas_inativas = []

        for origem, destino in grafo.edges:

            if (
                origem in nos_ativos
                and destino in nos_ativos
            ):
                arestas_ativas.append(
                    (origem, destino)
                )

            else:
                arestas_inativas.append(
                    (origem, destino)
                )

        # --------------------------------------------------
        # DESENHO
        # --------------------------------------------------

        plt.figure(figsize=(18, 12))

        # Arestas inativas primeiro
        nx.draw_networkx_edges(
            grafo,
            pos,
            edgelist=arestas_inativas,
            edge_color="gray",
            style="dashed",
            arrows=True,
            arrowsize=12,
            alpha=0.35
        )

        # Arestas ativas por cima
        nx.draw_networkx_edges(
            grafo,
            pos,
            edgelist=arestas_ativas,
            edge_color="black",
            width=2,
            arrows=True,
            arrowsize=15
        )

        # Nós
        nx.draw_networkx_nodes(
            grafo,
            pos,
            node_color=[
                cores[no]
                for no in grafo.nodes
            ],
            node_size=2000
        )

        # Labels
        nx.draw_networkx_labels(
            grafo,
            pos,
            labels=labels,
            font_size=8
        )

        # --------------------------------------------------
        # IDENTIFICAÇÃO DAS COLUNAS
        # --------------------------------------------------

        for coluna in range(1, prof_max + 1):

            plt.text(coluna, 1.5, f"P{coluna}", ha="center", fontsize=12, fontweight="bold")
        plt.text(0, 1.5, "INPUT", ha="center", fontsize=12, fontweight="bold")
        plt.text(prof_max + 1, 1.5, "OUTPUT", ha="center", fontsize=12, fontweight="bold")

        # --------------------------------------------------
        # CONFIGURAÇÃO
        # --------------------------------------------------

        plt.axis("off")

        plt.tight_layout()

        if salvar:
            plt.savefig(
                salvar,
                dpi=200,
                bbox_inches="tight"
            )

        plt.show()
        plt.close()
