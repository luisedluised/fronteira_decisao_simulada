cor_alvo,edgecolor_alvo,alpha_alvo = 'brown','crimson', 0.9
cor_base, edgecolor_base, alpha_base = 'dimgray','black', 0.6
cor_reta = 'black'
alpha_fundo, cor_fundo = 0.2, 'black'


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

import warnings
warnings.filterwarnings('ignore')

import random

def visualizar_classificacao(n, N, precisao, recall, ruido = 'automatico', distribuicao = 'normal',
                             expoentes = 'automatico', printar_info = False, skewness = 'media', figsize = (15, 8)):
    tentativas_por_coef = 40
    n_coefs_tentados = 100

    if type(ruido) not in (int, float):
        ruido = 0.15*((1/precisao))/(1+(2*recall)**2)
        
    if skewness == 'alta':
        skewness, e0 = 2, 1
    elif skewness == 'baixa':
        skewness,  e0 = 0.05, 0.5
    else: #media
        skewness, e0 = 0.5, 0.8
    
    if expoentes == 'automatico':
        expoentes = (
            np.min([np.max([e0 + skewness*np.random.normal(), 0.2]), 3])
         , np.min([np.max([e0 + skewness*np.random.normal(), 0.2]), 3])
        )
        
    if printar_info == True:
        print('expoentes', expoentes)

    n_classificados = round(recall*n/precisao)
    

    pontos = np.random.rand(N,2)
    
    if distribuicao == 'normal':
        pontos[:, 0] = [np.abs(1*np.random.normal() + 0.5) for x in np.arange(N)]
        pontos[:, 1] = [np.abs(1*np.random.normal() + 0.5) for x in np.arange(N)]

    pontos[:,1] = (pontos[:,1]**(expoentes[1]))
    pontos[:,0] = 1 - (pontos[:,0]**(expoentes[0]))

    a, b, tipo_fronteira = procurar_coef(n_classificados, pontos, tentativas_por_coef,n_coefs_tentados)
    if type(b) == str:
        return 'nao conseguiu encontrar a reta'

    pontos_classificados = emular_classificacao(recall, precisao, a, b, ruido,tipo_fronteira, pontos, n)


    _,f = plt.subplots(figsize = figsize)

    (x1, x2) = (-100, 100)
    y1 = a*x1 + b
    y2 = a*x2 + b
    
    plt.axvspan(-1000, 1000, facecolor=cor_fundo, alpha=alpha_fundo)
    fr_ = pontos_classificados[pontos_classificados.target == 0]
    f.scatter(fr_.x,fr_.y, color = cor_base, alpha =alpha_base, s=60, edgecolors  = edgecolor_base)
    fr_ = pontos_classificados[pontos_classificados.target == 1]
    f.scatter(fr_.x,fr_.y, alpha = alpha_alvo, s = 50, edgecolors  = edgecolor_alvo, color = cor_alvo)
    f.plot([x1,x2],[y1, y2], linewidth = 3, alpha = 0.7, color = cor_reta)

    margem = 0.01
    margem_quantil = 0
    if distribuicao == 'normal':
        margem_quantil = 0.001
        margem = 0.05
    quantil_inf = 0 + margem_quantil
    quantil_sup = 1 - margem_quantil
    
    xmin = pontos_classificados.x.quantile(quantil_inf) - margem
    xmax = pontos_classificados.x.quantile(quantil_sup) + margem
    
    ymin = pontos_classificados.y.quantile(quantil_inf) - margem
    ymax = pontos_classificados.y.quantile(quantil_sup) + margem
    
    f.set_xlim(xmin, xmax)
    f.set_ylim(ymin, ymax)
    
    
    plt.axis('off')
    f.axes.get_xaxis().set_visible(False)
    f.axes.get_yaxis().set_visible(False)

    TP = pontos_classificados[pontos_classificados.classificado == True][pontos_classificados.target == 1].shape[0]
    P =  pontos_classificados[pontos_classificados.target == 1].shape[0]
    recall_calculado = TP/P

    TP = pontos_classificados[pontos_classificados.classificado == True][pontos_classificados.target == 1].shape[0]
    C =  pontos_classificados[pontos_classificados.classificado == True].shape[0]
    precisao_calculada = TP/C
    
    if printar_info == True:
        print(round(recall_calculado, 2), round(precisao_calculada,2), 'recall, precisao')

    

def achar_reta(pontos, n_classificados, tentativas, coef):

    limites_possibilidades_b = [0,0,0,0]
    x_max, x_min = np.max(pontos[:,0]), np.min(pontos[:,0])
    y_max, y_min = np.max(pontos[:,1]), np.min(pontos[:,1])

    limites_possibilidades_b[0] = y_max - coef*x_max
    limites_possibilidades_b[1] = y_max - coef*x_min
    limites_possibilidades_b[2] = y_min - coef*x_max
    limites_possibilidades_b[3] = y_min - coef*x_min

    limite_inferior_b = np.min(limites_possibilidades_b)
    limite_superior_b = np.max(limites_possibilidades_b)

    bs = np.linspace(limite_inferior_b,limite_superior_b, tentativas)

    random.shuffle(bs)
    for b in bs:
        s_maior = 0
        s_menor = 0
        for ponto in pontos:
            x, y = ponto[0], ponto[1]
            s_maior += int(y >= coef*x + b)
            s_menor += int(y <= coef*x + b)
        if (s_maior > n_classificados*0.98) and  (s_maior < n_classificados*1.02):
            return b, 'acima_fronteira'
        if (s_menor > n_classificados*0.98) and  (s_menor < n_classificados*1.02):
            return b, 'abaixo_fronteira'
    return ('nao encontrou'), 'sem_fronteira'

def distancia_ponto_reta(x, y, a, b, c):
    return (np.abs(a*x + b*y + c)/((a**2+b**2)**(1/2)))

def procurar_coef(n_classificados, pontos, tentativas_por_coef, n_coefs_tentados, limites_coef = (-5,5)):
    v = np.linspace(limites_coef[0],limites_coef[1], int(n_coefs_tentados/2))
    v_ = 1/np.linspace(limites_coef[0],limites_coef[1], int(n_coefs_tentados/2))
    v = list(v) + list(v_)
    random.shuffle(v)
    for alfa in v:
        lin, tipo_fronteira = achar_reta(pontos, n_classificados, tentativas_por_coef, coef = alfa)
        if type(lin) != str:
            return alfa, lin, tipo_fronteira
    return alfa, lin, tipo_fronteira

def emular_classificacao(recall, precisao, a, b, ruido, tipo_fronteira, pontos, n):
    df = pd.DataFrame({'x':pontos[:, 0], 'y':pontos[:, 1]})
    if tipo_fronteira == 'acima_fronteira':
        df['classificado'] = df.y >= a*df.x + b
    elif tipo_fronteira == 'abaixo_fronteira':
        df['classificado'] = df.y <= a*df.x + b

    classificados = df[df.classificado == True]
    nao_classificados = df[df.classificado == False]

    nao_classificados['distancia_fronteira'] = nao_classificados.apply(lambda p: distancia_ponto_reta(x = p.x, y = p.y, a = -a, b = 1, c = -b), axis = 1)
    x_medio, y_medio = classificados.x.mean(), classificados.y.mean()
    nao_classificados['distancia_classificados'] = nao_classificados.apply(lambda p: ((p.x - x_medio)**2 + (p.y-y_medio)**2 )**(1/2), axis = 1)
    r = ruido*(np.random.rand(nao_classificados.shape[0]) +0)*(nao_classificados.distancia_classificados.max() - nao_classificados.distancia_classificados.min())
    r = ruido*(np.array([np.random.normal() for x in np.arange(nao_classificados.shape[0])]) -0.5)*(nao_classificados.distancia_classificados.max() - nao_classificados.distancia_classificados.min())

    nao_classificados['distancia'] = r + nao_classificados.distancia_classificados + nao_classificados.distancia_fronteira

    nao_classificados['distancia_classificados'] += r
    nao_classificados = nao_classificados.sort_values('distancia_classificados')

    TP = int(round(precisao*classificados.shape[0]))
    classificados['target'] = list(np.ones(TP)) + list(np.zeros(classificados.shape[0] - TP))

    target_nao_classificados = n - TP
    nao_classificados['target'] = list(np.ones(target_nao_classificados)) + list(np.zeros(nao_classificados.shape[0] - target_nao_classificados))

    return pd.concat([classificados,nao_classificados])


