import numpy as np
import cv2 as cv


class CGPEvolve:
    def __init__(self, cgp, executor):
        self.cgp = cgp
        self.executor = executor

    def erro(self, target, resultado):
        value = np.mean((target.astype(np.float64) - resultado.astype(np.float64)) ** 2)  # MSE

        return value

    def fitness_individuo(self,outputs,target,individuo):
        saida = individuo[-1][1]

        return self.erro(target,outputs[saida])

    def avaliar(self,individuo,inputs,outputs):
        grafo = self.cgp.construcao_grafo(individuo)
        if not self.cgp.verificacao_tipos(individuo[:-1], grafo):
            return (float("inf"),)

        erros = []

        for image_input_path, image_output_path in zip(inputs,outputs):
            img_output = cv.imread(image_output_path, cv.IMREAD_GRAYSCALE).astype(np.uint8)
            OUTPUTS = self.executor.executar(image_input_path,individuo,grafo)
            erro_img = self.fitness_individuo(OUTPUTS,img_output,individuo)
            erros.append(erro_img)
        fitness = np.mean(erros)

        return (fitness,)

    def mutacao(self, individuo):
        while True:
            mutate_rate_input, mutate_rate_node, mutate_rate_output = 0.5, 0.5, 0.5
            filho = type(individuo)(individuo)

            # Escolhe um gene de nó, ignorando o gene da saída
            indice = np.random.randint(0,len(filho) - 1)

            no = indice + 1
            grafo = self.cgp.construcao_grafo(filho)
            operadores_validos = []
            for operador in self.cgp.funcoes:
                conexoes = self.cgp.conexoes_validas(grafo, operador, no)

                if conexoes:
                    operadores_validos.append(operador)

            if np.random.random() < mutate_rate_node:
                operador = np.random.choice(operadores_validos)

                conexoes = self.cgp.conexoes_validas(grafo, operador, no)
                conexao = np.random.choice(conexoes)
                filho[indice] = [operador, conexao]

            else:
                if np.random.random() < mutate_rate_input:
                    continue 

                if np.random.random() < mutate_rate_output:
                    saida = np.random.randint(1, len(filho) - 1)
                    filho[-1] = ["out", saida]

            grafo_novo = self.cgp.construcao_grafo(filho)

            if self.cgp.verificacao_tipos(filho[:-1], grafo_novo):
                return filho,