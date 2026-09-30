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
        while True:
            mutate_rate_input, mutate_rate_node, mutate_rate_output = 0.5, 0.5, 0.5
            filho = type(individuo)(individuo)
            rnd = np.random.random()

            #PASSO A MAIS COMUM NA MUTAÇÃO: WHILE TRUE ATÉ TROCAR UM NÓ ATIVO
            #CUMULAÇÃO DE MUTAÇÃO X SEM CUMUÇÃO (RESET QUANDO TROCA UM NÃO-ATIVO)
            #levelsback

            #if rnd < mutate_rate_input:
            #    indice = np.random.randint(0, self.cgp.n_entradas)
            #    filho[indice] = [indice, indice]
            if rnd < mutate_rate_node:
                indice = np.random.randint(self.cgp.n_entradas, self.cgp.n_entradas + self.cgp.n_nos)
                no = indice            

                operadores_validos = []

                for operador in self.cgp.funcoes:
                    conexoes = self.cgp.conexoes_validas(individuo, operador, no)
                    if conexoes:
                        operadores_validos.append(operador)

                operador = np.random.choice(operadores_validos)
                conexoes = self.cgp.conexoes_validas(individuo, operador, no)
                conexao = np.random.choice(conexoes)

                filho[indice] = [operador,conexao]
            else:
                indice = np.random.randint(self.cgp.n_entradas + self.cgp.n_nos, len(individuo))
                no = np.random.randint(self.cgp.n_entradas, self.cgp.n_entradas + self.cgp.n_nos)
                filho[indice][1] = no

            if self.cgp.verificacao_tipos(filho):
                return filho,