import numpy as np
import cv2 as cv
from cgp import CGP
from cgpExecutor import CGPExecutor


class CGPEvolve:
    def __init__(self, cgp, executor, inputs, outputs):
        self.cgp = cgp
        self.executor = executor
        self.stats = {"nos_ativos":0, "tentativas":0, "descartadas":0, "validas":0, "mut_no":0, "mut_aresta":0, "mut_saida":0, "mut_ativa":0, "mut_inativa":0}
        assert len(inputs) == len(outputs), "quantidade de imagens e saidas deve ser igual"

        self.pares = []
        for inp, outp in zip(inputs, outputs):
            img = cv.imread(inp, cv.IMREAD_GRAYSCALE)
            gt = cv.imread(outp, cv.IMREAD_GRAYSCALE)
            if img.shape != gt.shape:
                raise ValueError(f"Imagem de entrada {inp} e ground truth {outp} devem ter o mesmo tamanho")
            if img is None or gt is None:
                raise ValueError(f"Erro ao ler as imagens {inp} ou {outp}")
            self.pares.append((inp, img.astype(np.uint8), gt.astype(np.uint8)))
    
    def erro(self, target, resultado):
        if resultado.shape != target.shape:
            return 0.0
        value = np.mean((target.astype(np.float64) - resultado.astype(np.float64)) ** 2)  # MSE
        value = max(value, 1e-10)  
        return float(10 * np.log10((255**2) / value))  # PSNR

    def fitness_individuo(self,outputs,target,individuo):
        erros = []
        for n in range(self.cgp.n_entradas, self.cgp.n_saidas + 1):
            saida = individuo[-n][1]
            erros.append(self.erro(target,outputs[saida]))
        return erros[0] if len(erros) == 1 else np.mean(erros)

    def avaliar(self,individuo):
        self.cgp.genoma = individuo

        if not self.cgp.verificacao_tipos(individuo):
            return (-float("inf"),)

        nos_ativos = self.executor.nos_ativos()
        erros = []
        for _, img, gt in self.pares:
            OUTPUTS = self.executor.executar_array(img, nos_ativos)
            erros.append(self.fitness_individuo(OUTPUTS, gt, individuo))
        return erros
    
    def fitness_medio(self, individuo):
        erros = self.avaliar(individuo)
        fitness = np.mean(erros)
        return (fitness,)

    def mutacao(self, individuo):
        mutate_active_node = False
        filho = CGP(self.cgp.n_entradas, self.cgp.funcoes, self.cgp.n_nos, self.cgp.n_saidas, self.cgp.levels_back, self.cgp.nos_por_camada)
        filho.genoma = type(individuo)([gene.copy() for gene in individuo])
        fe = CGPExecutor(filho)
        
        while not mutate_active_node:
            nos_ativos = fe.nos_ativos()
            nos_inativos = [i for i in range(filho.n_entradas, filho.n_entradas + filho.n_nos) if i not in nos_ativos]
            self.stats["nos_ativos"] = len(nos_ativos)
            rnd = np.random.random()
            self.stats["tentativas"] += 1

            #PASSO A MAIS COMUM NA MUTAÇÃO: WHILE TRUE ATÉ TROCAR UM NÓ ATIVO
            #CUMULAÇÃO DE MUTAÇÃO X SEM CUMULAÇÃO (RESET QUANDO TROCA UM NÃO-ATIVO) [ESTÁ CUMULANDO MUTACOES]
    
            #if rnd < 1/4: #MUTAÇÃO DOS NÓS DE ENTRADA 
            #    indice = np.random.randint(0, filho.n_entradas)
            #    filho.genoma[indice] = [indice, indice]
            if rnd < 1/3:
                self.stats["mut_no"] += 1
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
                self.stats["mut_aresta"] += 1
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
                self.stats["mut_saida"] += 1
                ### MUTAÇÃO DAS SAÍDAS ###
                indice = np.random.randint(filho.n_entradas + filho.n_nos, filho.n_entradas + filho.n_nos + filho.n_saidas)
                no_atual = filho.genoma[indice][1]
                no = np.random.randint(filho.n_entradas, filho.n_entradas + filho.n_nos)
                if no != no_atual:
                    filho.genoma[indice][1] = no
                mutate_active_node = no not in nos_ativos
                

            if mutate_active_node and filho.verificacao_tipos(filho.genoma):
                self.stats["validas"] += 1
                self.stats["mut_ativa"] += 1
                self.stats["nos_ativos"] = len(self.executor.nos_ativos())
                return filho.genoma,
            if not mutate_active_node and filho.verificacao_tipos(filho.genoma):
                self.stats["validas"] += 1
                self.stats["mut_inativa"] += 1
                self.stats["nos_ativos"] = len(self.executor.nos_ativos())
            if not filho.verificacao_tipos(filho.genoma):
                self.stats["descartadas"] += 1
                
            mutate_active_node = False