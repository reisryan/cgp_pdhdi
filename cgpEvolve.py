import numpy as np
import cv2 as cv
from cgp import CGP
from cgpExecutor import CGPExecutor


class CGPEvolve:
    def __init__(self, cgp, executor):
        self.cgp = cgp
        self.executor = executor

    def erro(self, target, resultado):
        N, M = target.shape[:2]
        value = np.mean((target.astype(np.float64) - resultado.astype(np.float64)) ** 2)  # MSE
        return 10* np.log10((255**2)/value) #PNSR

    def fitness_individuo(self,outputs,target,individuo):
        erros = []
        for n in range(1, self.cgp.n_saidas + 1):
            saida = individuo[-n][1]
            erros.append(self.erro(target,outputs[saida]))
        return erros[0] if len(erros) == 1 else np.mean(erros)

    def avaliar(self,individuo,inputs,outputs):
        self.cgp.genoma = individuo

        if not self.cgp.verificacao_tipos(individuo):
            return (float("inf"),)

        erros = []

        for image_input_path, image_output_path in zip(inputs,outputs):
            img_output = cv.imread(image_output_path, cv.IMREAD_GRAYSCALE).astype(np.uint8)
            OUTPUTS = self.executor.executar(image_input_path)
            erro_img = self.fitness_individuo(OUTPUTS,img_output,individuo)
            erros.append(erro_img)
        fitness = np.mean(erros)

        return (fitness,)

    def mutacao(self, individuo):
        mutate_active_node = False
        filho = CGP(self.cgp.n_entradas, self.cgp.funcoes, self.cgp.n_nos, self.cgp.n_saidas, self.cgp.levels_back, self.cgp.nos_por_camada)
        filho.genoma = type(individuo)([gene.copy() for gene in individuo])
        fe = CGPExecutor(filho)

        while not mutate_active_node:
            nos_ativos = fe.nos_ativos()
            rnd = np.random.random()

            #PASSO A MAIS COMUM NA MUTAÇÃO: WHILE TRUE ATÉ TROCAR UM NÓ ATIVO
            #CUMULAÇÃO DE MUTAÇÃO X SEM CUMULAÇÃO (RESET QUANDO TROCA UM NÃO-ATIVO)
    
            #if rnd < 1/4: #MUTAÇÃO DOS NÓS DE ENTRADA 
            #    indice = np.random.randint(0, filho.n_entradas)
            #    filho.genoma[indice] = [indice, indice]
            if rnd < 1/3:
                ### MUTAÇÃO DAS FUNÇÕES ###
                no = np.random.randint(filho.n_entradas, filho.n_entradas + filho.n_nos)
                           
                operadores_validos = []

                for operador in filho.funcoes:
                    conexoes = filho.conexoes_validas(filho.genoma, operador, no)
                    if conexoes and operador != filho.genoma[no][0]:
                        operadores_validos.append(operador)

                if operadores_validos:
                    operador = np.random.choice(operadores_validos)
                    filho.genoma[no][0] = operador
                    mutate_active_node = no in nos_ativos
            elif rnd < 2/3:
                ### MUTAÇÃO DAS ARESTAS ###
                no = np.random.randint(filho.n_entradas, filho.n_entradas + filho.n_nos)
                conexoes = filho.conexoes_validas(filho.genoma, filho.genoma[no][0], no)
                if filho.genoma[no][1] in conexoes:
                    conexoes.remove(filho.genoma[no][1]) #.observar

                if conexoes:
                    conexao = np.random.choice(conexoes)
                    filho.genoma[no][1] = conexao
                    mutate_active_node = no in nos_ativos
            else:
                ### MUTAÇÃO DAS SAÍDAS ###
                indice = np.random.randint(filho.n_entradas + filho.n_nos, filho.n_entradas + filho.n_nos + filho.n_saidas)
                no_atual = filho.genoma[indice][1]
                no = np.random.randint(filho.n_entradas, filho.n_entradas + filho.n_nos)
                if no != no_atual:
                    filho.genoma[indice][1] = no
                mutate_active_node = no not in nos_ativos
                

            if mutate_active_node and filho.verificacao_tipos(filho.genoma):
                return filho.genoma,
            mutate_active_node = False