import cv2 as cv


class CGPExecutor:
    def __init__(self, cgp):
        self.cgp = cgp

    def nos_ativos(self):
        nodes_enabled = set()

        to_verify = []
        for i in range(self.cgp.n_saidas):
            to_verify.append(self.cgp.genoma[-(i+1)][1])

        while to_verify:
            node_id = to_verify.pop()
            if node_id in nodes_enabled:
                continue

            nodes_enabled.add(node_id)
            if node_id < self.cgp.n_entradas:
                continue

            gene = self.cgp.genoma[node_id]
            conexao = gene[1]
            to_verify.append(conexao)
        return nodes_enabled

    def view_structure(self, genoma):
        self.cgp.genoma = genoma
        nodes_enabled = self.nos_ativos()
        
        for i in range(self.cgp.n_entradas, self.cgp.n_entradas + self.cgp.n_nos):
            if i not in nodes_enabled:
                continue
            gene = genoma[i]
            operador_id = gene[0]
            conexao = gene[1]
            print(f"Nó {i}: Operador {operador_id}, Conexão {conexao}")
            print(f"Função: {self.cgp.funcoes[operador_id]["funcao"].__name__}")
        print(f"Saídas: {genoma[-self.cgp.n_saidas:]}")


    """def executar(self, arquivo): #grafo
        OUTPUTS = {}

        OUTPUTS[0] = cv.imread(arquivo, cv.IMREAD_GRAYSCALE).astype("uint8")
        nodes_ativos = self.nos_ativos()

        for no in range(self.cgp.n_entradas,self.cgp.n_entradas+self.cgp.n_nos):
            if (no < self.cgp.n_entradas) or (no not in nodes_ativos):
                continue

            gene = self.cgp.genoma[no]
            operador_id = gene[0]
            
            funcao = self.cgp.funcoes[operador_id]["funcao"]
            conexao = gene[1]
            entrada = OUTPUTS[conexao]
            OUTPUTS[no] = funcao(entrada)

        return OUTPUTS"""

    def executar_array(self, imagem, nos_ativos=None):
        if nos_ativos is None:
            nos_ativos = self.nos_ativos()

        OUTPUTS = {i: imagem for i in range(self.cgp.n_entradas)}

        for no in range(self.cgp.n_entradas, self.cgp.n_entradas + self.cgp.n_nos):
            if no not in nos_ativos:
                continue
            op, con = self.cgp.genoma[no]
            funcao = self.cgp.funcoes[op]["funcao"]
            OUTPUTS[no] = funcao(OUTPUTS[con])

        return OUTPUTS