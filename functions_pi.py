import cv2 as cv
from skimage.filters import threshold_multiotsu, threshold_sauvola
import numpy as np
from cgp import Imagem, ImagemBinaria

#ALGORITMO MEMÉTICO NO PARÂMETRO DAS FUNÇÕES
# A CADA GERAÇÃO PEGAR O ELITE -> MELHOR FILHO X PAI, E OTIMIZA OS PARÂMETROS

#BFGS -: 
def clahe(imagem):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(imagem)

def limiares(imagem):
    lim = threshold_multiotsu(imagem, classes=3)
    return lim

def otsu(imagem):
    _, imgbin = cv.threshold(imagem, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return imgbin

def otsu_method(hist):
    num_bins = hist.shape[0]
    total = hist.sum()
    sum_total = np.dot(np.arange(num_bins), hist)
    weight_background = 0.0
    sum_background = 0.0
    optimum_value = 0
    maximum = -np.inf
    for t in range(num_bins):
        weight_background += hist[t]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]
        mean_foreground = (
            sum_total - sum_background
        ) / weight_foreground
        mean_background = sum_background / weight_background

        variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )

        if variance > maximum:
            optimum_value = t
            maximum = variance

    return optimum_value, maximum

def TSMO(imagem):
    """Modified Two-Stage Multithreshold Otsu Method."""

    M, L = 32, 256
    hist = np.bincount(imagem.ravel(), minlength=L)
    N = L // M

    # Histograma agrupado e normalizado
    norm_hist = np.zeros(M, dtype=np.float32)

    for i in range(M):
        norm_hist[i] = hist[i * N:(i + 1) * N].sum()

    if norm_hist.max() > 0:
        norm_hist = (norm_hist / norm_hist.max()) * 100

    # Estimativa dos vales
    probs = np.zeros(M, dtype=int)
    costs = np.zeros(M, dtype=float)

    for i in range(1, M - 1):
        if norm_hist[i] > norm_hist[i - 1] or norm_hist[i] > norm_hist[i + 1]:
            probs[i] = 0
        elif norm_hist[i] < norm_hist[i - 1] and norm_hist[i] == norm_hist[i + 1]:
            probs[i] = 1
            costs[i] = norm_hist[i - 1] - norm_hist[i]
        elif norm_hist[i] == norm_hist[i - 1] and norm_hist[i] < norm_hist[i + 1]:
            probs[i] = 3
            costs[i] = norm_hist[i + 1] - norm_hist[i]
        elif norm_hist[i] < norm_hist[i - 1] and norm_hist[i] < norm_hist[i + 1]:
            probs[i] = 4
            costs[i] = norm_hist[i - 1] + norm_hist[i + 1] - 2 * norm_hist[i]
        elif norm_hist[i] == norm_hist[i - 1] and norm_hist[i] == norm_hist[i + 1]:
            probs[i] = probs[i - 1]
            costs[i] = probs[i - 1]

    for i in range(1, M - 1):
        if probs[i] != 0:
            probs[i] = (probs[i - 1] + probs[i] + probs[i + 1]) // 4

    valleys = np.where(probs > 0)[0].tolist()

    # Otsu nas regiões dos vales
    thresholds = []

    for valley in valleys:
        start_pos = max(0, valley * N - N)
        end_pos = min(L, (valley + 2) * N)
        regional_hist = hist[start_pos:end_pos]

        if regional_hist.size == 0 or regional_hist.sum() == 0:
            continue

        sub_threshold, variance = otsu_method(regional_hist)
        thresholds.append((start_pos + sub_threshold, variance))

       # Ordena os limiares pela intensidade, em ordem crescente.
    thresholds.sort(key=lambda x: x[0])
    limiares = [int(t) for t, _ in thresholds]

    # TO1: menor limiar; TO2: segundo menor limiar.
    if len(limiares) >= 2:
        TO1 = limiares[0]
        TO2 = limiares[1]
    elif len(limiares) == 1:
        TO1 = limiares[0]
        TO2 = TO1
    else:
        TO1, _ = cv.threshold(
            imagem, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )
        TO2 = int(TO1)

    # A função retorna a imagem binarizada por TO2 e os limiares.
    _, imagem_binarizada = cv.threshold(
        imagem, TO2, 255, cv.THRESH_BINARY
    )

    return imagem_binarizada

def multiotsu(imagem):
    #lim = limiares(imagem)
    _, imgbin = cv.threshold(imagem, 5, 255, cv.THRESH_BINARY)
    return imgbin

def nick(imagem):
    ws = 7
    k_nick = -0.05  # Parâmetro de controle da influência do desvio padrão local.
    img_f = imagem.astype(np.float64)
    
    # Cálculo eficiente da média local m(x, y) usando Box Filter (filtro de caixa 2D uniforme).
    m_img = cv.boxFilter(img_f, -1, (ws, ws), normalize=True)
    
    # Cálculo da média dos quadrados locais para computar a variância local de forma otimizada.
    img_sq = img_f ** 2
    sum_sq_img = cv.boxFilter(img_sq, -1, (ws, ws), normalize=False)
    
    NP = ws * ws
        # Cálculo do radicando (estimador de variância/desvio padrão da vizinhança).
        # Garante-se que valores levemente negativos causados por precisão numérica sejam zerados.
    radicand = (sum_sq_img - (m_img ** 2)) / NP
    radicand = np.maximum(radicand, 0.0)
    
        # Geração do mapa bidimensional de limiares locais T_nick_img.
    T_nick_img = m_img + k_nick * np.sqrt(radicand)
    
        # Segmentação local: pixels com intensidade menor ou igual ao limiar local viram preto (0),
        # e pixels maiores viram branco (255).
    img_nick = np.zeros_like(imagem, dtype=np.uint8)
    img_nick[imagem <= T_nick_img] = 0
    img_nick[imagem > T_nick_img] = 255
    
        # ==============================================================================
        # 5. DETECÇÃO DE MANCHAS (SMEAR) POR DENSIDADE E MESCLAGEM HÍBRIDA
        # ==============================================================================
        # Manchas ou vazamento de tinta (smear) degradam a binarização global.
        # Este bloco divide a imagem binarizada global em blocos não-sobrepostos,
        # calcula a densidade de pixels pretos (texto/mancha) em cada bloco e
        # rotula blocos com densidade anormalmente alta (outliers) como manchas.
    
    k_smear = 0.001
    h, w = imagem.shape
    fSlist = []
    blockscoords = []
    
        # Partição da imagem em blocos disjuntos de dimensão ws x ws.
    for y in range(0, h, ws):
        for x in range(0, w, ws):
            block = imagem[y:y+ws, x:x+ws]
            tp = block.size
            if tp > 0:
                # Quantidade de pixels pretos no bloco.
                bp = np.sum(block == 0)
                fS = bp / tp                 # Fração de pixels pretos (densidade).
                fSlist.append(fS)
                blockscoords.append((y, x, fS, block.shape))
    
        # Média e desvio padrão global da densidade de pixels pretos por bloco.
    m_fS, s_fS = np.mean(fSlist), np.std(fSlist)
    
    smear_mask = np.zeros_like(imagem)
    
        # Classificação dos blocos: se a fração de pretos exceder (média + k_smear * desvio_padrão),
        # a região é classificada como mancha ou região densa de interesse.
    for y, x, fS, shape in blockscoords:
        if fS > (m_fS + k_smear * s_fS):
            smear_mask[y:y+shape[0], x:x+shape[1]] = 255
    
        # Dilatação morfológica da máscara de manchas para abranger as bordas e áreas de transição.
    kernel_dilate_mask = np.ones((ws, ws), np.uint8)
    smear_mask = cv.dilate(smear_mask, kernel_dilate_mask, iterations=1)
    
        # Mesclagem híbrida (Hybrid Binarization):
        # - Nas áreas cobertas pela máscara de mancha (smear_mask == 255), utiliza-se o resultado
        #   do algoritmo local adaptativo (img_nick), que é mais robusto a variações locais de iluminação.
        # - Nas áreas limpas de fundo (fundo normal), preserva-se o resultado da binarização global.
    img_b2 = np.where(smear_mask == 255, img_nick, imagem)
    return img_b2

def sauvola(imagem):
    ws = 17
    thresh_sauvola = threshold_sauvola(imagem, window_size=ws, k=0.2)
    img_sauvola = (imagem > thresh_sauvola).astype("uint8") * 255
    return img_sauvola

def open_morphology(imagem):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    return cv.morphologyEx(imagem, cv.MORPH_OPEN, kernel)

def close_morphology(imagem):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    return cv.morphologyEx(imagem, cv.MORPH_CLOSE, kernel)

def square_root(imagem):
    return (cv.sqrt((imagem / 255.0).astype(np.float32)) * 255).astype(np.uint8)

def square(imagem):
    return (cv.pow((imagem / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)

def f_exp(imagem):
	return (cv.exp((imagem / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)

def f_log(imagem):
	return np.log1p(imagem).astype(np.uint8)

def median_blur(imagem):
    return cv.medianBlur(imagem.astype(np.uint8), 3)

def gaussian_blur(imagem):
    return cv.GaussianBlur(imagem.astype(np.uint8), (5, 5), 0)

def laplacian(imagem):
    laplacian_64f = cv.Laplacian(imagem, cv.CV_64F)
    abs_laplacian = np.absolute(laplacian_64f)
    return np.uint8(np.clip(abs_laplacian, 0, 255))

def sobel(imagem):
    sobel_x = cv.Sobel(imagem, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(imagem, cv.CV_64F, 0, 1, ksize=5)
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    sobel_final = cv.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    return cv.convertScaleAbs(sobel_final)

def erode(imagem):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    return cv.erode(imagem, kernel)

def dilate(imagem):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    return cv.dilate(imagem, kernel)

def normalize(imagem):
    return cv.normalize(imagem, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

def f_invert(imagem):
    return cv.bitwise_not(imagem)

def bilateral_filter(imagem):
    # d=5 (vizinhança), sigmaColor=75, sigmaSpace=75 
    return cv.bilateralFilter(imagem.astype(np.uint8), 5, 75, 75)

def morph_gradient(imagem):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    return cv.morphologyEx(imagem, cv.MORPH_GRADIENT, kernel)

def f_sharpen(imagem):
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    return cv.filter2D(imagem, -1, kernel)

