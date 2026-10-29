import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import glob
import ast
import re

CORES = [
    "green", "darkorange", "royalblue", "purple", "crimson",
    "teal", "brown", "magenta", "olive", "black"
]


def ler_arquivo(caminho):
    dados = {}
    with open(caminho, "r", encoding="utf-8") as arquivo:
        conteudo = arquivo.read()

    blocos = re.split(r"\n\s*\n", conteudo.strip())

    for bloco in blocos:
        item = {}
        for linha in bloco.splitlines():
            if ":" not in linha:
                continue

            chave, valor = linha.split(":", 1)
            chave = chave.strip()
            valor = valor.strip()

            if chave in (
                "iteracao", "geracoes", "lambda", "nos",
                "nos_total", "fitness_final", "historico_pais",
                "historico_lambda", "historico_stats"
            ):
                try:
                    item[chave] = ast.literal_eval(valor)
                except (ValueError, SyntaxError):
                    item[chave] = valor

        if "iteracao" in item:
            numero = int(item["iteracao"])
            dados[numero] = item

    return dados


def localizar_arquivos(imagem):
    arquivos = glob.glob("*_result.txt")

    if imagem.lower() == "all":
        return sorted(arquivos)

    nome = os.path.splitext(os.path.basename(imagem))[0]
    nome = re.sub(r"_estGT_dibco2013$", "", nome)
    nome = re.sub(r"_dibco2013$", "", nome)

    encontrados = []
    for caminho in arquivos:
        identificador = os.path.basename(caminho).replace("_result.txt", "")
        identificador = re.sub(r"_estGT_dibco2013$", "", identificador)
        identificador = re.sub(r"_dibco2013$", "", identificador)

        if identificador.lower() == nome.lower():
            encontrados.append(caminho)

    return encontrados


def carregar_dados(imagem):
    arquivos = localizar_arquivos(imagem)

    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo *_result.txt encontrado para '{imagem}'."
        )

    dados = []
    for caminho in arquivos:
        iteracoes = ler_arquivo(caminho)
        for numero, item in iteracoes.items():
            item["_imagem"] = caminho.replace("_result.txt", "")
            item["_iteracao"] = numero
            dados.append(item)

    if not dados:
        raise ValueError("Os arquivos não contêm históricos de iteração válidos.")

    return dados


def obter_iteracoes(dados):
    return sorted({item["_iteracao"] for item in dados})


def obter_historico_pais(item):
    return np.asarray(item.get("historico_pais", []), dtype=float)


def obter_historico_filhos(item):
    historico = item.get("historico_lambda", [])
    resultado = []

    for geracao in historico:
        valores = np.asarray(geracao, dtype=float).flatten()
        valores = valores[np.isfinite(valores)]
        resultado.append(valores)

    return resultado


def agregar_pais(dados, iteracao):
    historicos = [
        obter_historico_pais(item)
        for item in dados
        if item["_iteracao"] == iteracao
    ]
    historicos = [h for h in historicos if len(h) > 0]

    if not historicos:
        return None

    tamanho = min(len(h) for h in historicos)
    matriz = np.vstack([h[:tamanho] for h in historicos])

    geracoes = np.arange(tamanho)
    media = np.mean(matriz, axis=0)
    desvio = np.std(matriz, axis=0)

    return geracoes, media, desvio, len(historicos)


def agregar_filhos(dados, iteracao):
    historicos = [
        obter_historico_filhos(item)
        for item in dados
        if item["_iteracao"] == iteracao
    ]
    historicos = [h for h in historicos if h]

    if not historicos:
        return None

    tamanho = min(len(h) for h in historicos)
    medias = []
    minimos = []
    maximos = []

    for g in range(tamanho):
        valores = [
            filho
            for historico in historicos
            for filho in historico[g]
            if np.isfinite(filho)
        ]

        if not valores:
            medias.append(np.nan)
            minimos.append(np.nan)
            maximos.append(np.nan)
        else:
            medias.append(float(np.mean(valores)))
            minimos.append(float(np.min(valores)))
            maximos.append(float(np.max(valores)))

    geracoes = np.arange(1, tamanho + 1)

    return (
        geracoes,
        np.asarray(medias),
        np.asarray(minimos),
        np.asarray(maximos),
        len(historicos)
    )


def grafico1(params):
    dados, nome_arquivo = params
    iteracoes = obter_iteracoes(dados)

    plt.figure(figsize=(11, 6))

    for indice, iteracao in enumerate(iteracoes):
        resultado = agregar_pais(dados, iteracao)
        if resultado is None:
            continue

        geracoes, media, desvio, quantidade = resultado
        cor = CORES[indice % len(CORES)]

        plt.plot(
            geracoes, media, color=cor, linewidth=2,
            label=f"It. {iteracao} — pais (n={quantidade} imagens)"
        )
        plt.fill_between(
            geracoes, media - desvio, media + desvio,
            color=cor, alpha=0.20
        )

    plt.xlabel("Geração")
    plt.ylabel("Fitness médio")
    plt.title("Convergência dos pais")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{nome_arquivo}_pais.png", dpi=200, bbox_inches="tight")
    plt.close()


def grafico2(params):
    dados, nome_arquivo = params
    iteracoes = obter_iteracoes(dados)

    plt.figure(figsize=(11, 6))

    for indice, iteracao in enumerate(iteracoes):
        resultado = agregar_filhos(dados, iteracao)
        if resultado is None:
            continue

        geracoes, media, minimo, maximo, quantidade = resultado
        cor = CORES[indice % len(CORES)]

        plt.plot(
            geracoes, media, color=cor, linewidth=2,
            label=f"It. {iteracao} — média dos filhos"
        )
        plt.fill_between(
            geracoes, minimo, maximo, color=cor, alpha=0.20,
            label=f"It. {iteracao} — mínimo–máximo"
        )

    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.title("Fitness dos filhos por geração")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{nome_arquivo}_filhos.png", dpi=200, bbox_inches="tight")
    plt.close()


def obter_estatistica_nos(item, no_total=30):
    historico = item.get("historico_stats", [])
    ativos = []
    inativos = []

    total_nos = no_total

    for stats in historico:
        if not isinstance(stats, dict):
            ativos.append(np.nan)
            inativos.append(np.nan)
            continue

        numero_ativos = stats.get("nos_ativos")
        numero_inativos = no_total - numero_ativos if numero_ativos is not None else None
        

        if numero_ativos is None:
            ativos.append(np.nan)
            inativos.append(np.nan)
            continue

        ativos.append(float(numero_ativos))

        if numero_inativos is not None:
            inativos.append(float(numero_inativos))
        #elif total_nos is not None:
        #    inativos.append(float(total_nos) - float(numero_ativos))
        else:
            inativos.append(np.nan)

    return np.asarray(ativos), np.asarray(inativos)


def grafico3(params):
    dados, nome_arquivo, modo = params

    if modo == "mean":
        selecionados = dados
        titulo_modo = "Média"
    else:
        numero_iteracao = int(modo.replace("iter", "").strip())
        selecionados = [
            item for item in dados
            if item["_iteracao"] == numero_iteracao
        ]
        if not selecionados:
            raise ValueError(f"Iteração {numero_iteracao} não encontrada.")
        titulo_modo = f"Iteração {numero_iteracao}"

    series_ativos = []
    series_inativos = []

    for item in selecionados:
        ativos, inativos = obter_estatistica_nos(item)
        if len(ativos):
            series_ativos.append(ativos)
            series_inativos.append(inativos)

    if not series_ativos:
        raise ValueError("Não foram encontrados dados de nós em historico_stats.")

    tamanho = min(len(s) for s in series_ativos)
    matriz_ativos = np.vstack([s[:tamanho] for s in series_ativos])
    matriz_inativos = np.vstack([s[:tamanho] for s in series_inativos])

    media_ativos = np.nanmean(matriz_ativos, axis=0)
    media_inativos = np.nanmean(matriz_inativos, axis=0)
    geracoes = np.arange(1, tamanho + 1)

    plt.figure(figsize=(11, 6))
    plt.plot(geracoes, media_ativos, color="green", linewidth=2,
             label="Nós ativos")
    plt.plot(geracoes, media_inativos, color="royalblue", linewidth=2,
             label="Nós inativos")
    plt.xlabel("Geração")
    plt.ylabel("Quantidade média de nós")
    plt.title(f"Nós ativos e inativos por geração — {titulo_modo}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{nome_arquivo}_nos.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "graphic":
        print("Uso:")
        print("  python3 data_analysis.py graphic all all")
        print("  python3 data_analysis.py graphic imagem_nome all")
        print("  python3 data_analysis.py graphic 3 imagem_nome mean")
        print("  python3 data_analysis.py graphic 3 imagem_nome iter 2")
        return

    if sys.argv[2] == "3":
        if len(sys.argv) < 5:
            print("Uso: python3 data_analysis.py graphic 3 imagem_nome <mean|iter n>")
            return

        imagem = sys.argv[3]
        if sys.argv[4] == "iter":
            if len(sys.argv) < 6:
                print("Informe o número da iteração.")
                return
            modo = f"iter {sys.argv[5]}"
        else:
            modo = sys.argv[4]

        try:
            dados = carregar_dados(imagem)
            nome = "all" if imagem.lower() == "all" else os.path.splitext(imagem)[0]
            grafico3((dados, nome, modo))
            print(f"Gráfico salvo: {nome}_nos.png")
        except (FileNotFoundError, ValueError) as erro:
            print(f"Erro: {erro}")
        return

    imagem = sys.argv[2]

    try:
        dados = carregar_dados(imagem)
        nome = "all" if imagem.lower() == "all" else os.path.splitext(imagem)[0]

        grafico1((dados, nome))
        grafico2((dados, nome))

        print(f"Gráficos salvos: {nome}_pais.png e {nome}_filhos.png")

    except (FileNotFoundError, ValueError) as erro:
        print(f"Erro: {erro}")


if __name__ == "__main__":
    main()