import nltk
from xml.etree import ElementTree as ET
import os
import math
import time
import csv
from ast import literal_eval
from collections import defaultdict
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, average_precision_score
import ast

try:
    start_time = time.time()
    print("\n=================================================================================================\n")
    print("1 - Configurando diretórios e carregando dados...\n")
    dir_atual = os.getcwd()
    dir_avalia = os.path.join(dir_atual, 'AVALIA')
    if not os.path.exists(dir_avalia):
        os.makedirs(dir_avalia)

    arquivo_resultados = os.path.join(dir_atual, 'RESULT', 'RESULTADOS.csv')
    arquivo_resultados_esperados = os.path.join(dir_atual, 'RESULT', 'expectedInstruction.csv')
    dados_resultados = pd.read_csv(arquivo_resultados, delimiter=';', header=None)
    dados_esperados = pd.read_csv(arquivo_resultados_esperados, delimiter=';', header=None)

    print("2 - Carregando configurações e inicializando stemmer...\n")
    arquivo_config = os.path.join(dir_atual, 'config', 'config.txt')
    escolha_stemmer = "STEMMER"
    if os.path.exists(arquivo_config):
        with open(arquivo_config, 'r') as file:
            escolha_stemmer = file.readline().strip()

    stemmer_ativo = True if escolha_stemmer == "STEMMER" else False
    stemmer = PorterStemmer() if stemmer_ativo else None

    print("3 - Pré-processando os resultados...\n")
    dados_resultados['Processados'] = dados_resultados[1].apply(
        lambda texto: [(0, 0, stemmer.stem(str(score))) for score in ast.literal_eval(texto)]
        if stemmer_ativo and isinstance(ast.literal_eval(texto), list) else 
        [(0, 0, str(score)) for score in ast.literal_eval(texto)]
        if isinstance(ast.literal_eval(texto), list) else
        [(0, 0, stemmer.stem(str(texto))) if stemmer_ativo else (0, 0, str(texto))]
    )

    print("4 - Organizando dados para avaliação e agrupando...\n")
    resultados_organizados = dados_resultados.explode('Processados').reset_index(drop=True)
    resultados_organizados[['Posicao', 'NumDoc', 'ScoreDoc']] = pd.DataFrame(resultados_organizados['Processados'].tolist(), index=resultados_organizados.index)
    resultados_organizados = resultados_organizados.drop(['Processados', 0, 1], axis=1)
    resultados_agrupados = resultados_organizados.groupby('Posicao')

    print("5 - Renomeando colunas, removendo cabeçalhos e convertendo tipos...\n")
    dados_esperados.columns = ['NumeroQuery', 'Esperado', 'ScoreDoc']
    dados_esperados = dados_esperados.iloc[1:]
    dados_esperados['NumeroQuery'] = dados_esperados['NumeroQuery'].astype('int64')

    print("6 - Inicializando listas para armazenar métricas...\n")
    lista_precision_recall = []
    lista_f1 = []
    lista_precision5 = []
    lista_precision10 = []
    lista_rPrecision = []
    lista_mapScore = []

    print("7 - Percorrendo grupos...\n")
    for query, grupo in resultados_agrupados:
        if query in dados_esperados['NumeroQuery'].values:
            esperados_grupo = dados_esperados[dados_esperados['NumeroQuery'] == query]['ScoreDoc']
            if len(grupo) > 0 and len(esperados_grupo) > 0:
                precision_recall = precision_recall_curve(esperados_grupo, grupo['ScoreDoc'])
                f1 = f1_score(esperados_grupo, grupo['ScoreDoc'])
                precision5 = precision_score(esperados_grupo, grupo['ScoreDoc'], pos_label=1, average='binary', k=5)
                precision10 = precision_score(esperados_grupo, grupo['ScoreDoc'], pos_label=1, average='binary', k=10)
                rPrecision = precision_score(esperados_grupo, grupo['ScoreDoc'])
                mapScore = average_precision_score(esperados_grupo, grupo['ScoreDoc'])

                lista_precision_recall.append(precision_recall)
                lista_f1.append(f1)
                lista_precision5.append(precision5)
                lista_precision10.append(precision10)
                lista_rPrecision.append(rPrecision)
                lista_mapScore.append(mapScore)
            else:
                print(f"Dados insuficientes para cálculos métricos na query: {query}.")

    print("8 - Calculando médias das métricas...\n")
    if len(lista_precision_recall) > 0 :
        media_precision_recall = np.mean([pr[0] for pr in lista_precision_recall], axis=0)
        if isinstance(media_precision_recall, np.ndarray) and len(media_precision_recall) > 0:
            with open('RELATORIO.MD', 'w') as relatorio:
                relatorio.write("# Avaliação do Sistema de Recuperação de Informação\n\n")
                relatorio.write("## Resultados com Stemmer\n")
                relatorio.write(f"- **F1 Score:** {np.mean(lista_f1)}\n")
                relatorio.write(f"- **Precision@5:** {np.mean(lista_precision5)}\n")
                relatorio.write(f"- **Precision@10:** {np.mean(lista_precision10)}\n")
                relatorio.write(f"- **R-Precision:** {np.mean(lista_rPrecision)}\n")
                relatorio.write(f"- **MAP:** {np.mean(lista_mapScore)}\n")

            plt.plot(media_precision_recall[1], media_precision_recall[0], marker='o')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Curva Precision-Recall de 11 pontos')
            plt.savefig(f'11pontos-{escolha_stemmer.lower()}.pdf')
            plt.show()
        else:
            print("Não há dados suficientes para gerar a curva de precisão-recall.")
    else:
        print("Não há dados suficientes para calcular as métricas.")

    print("9 - Salvando os dados...\n")
    resultados_organizados.to_csv(f'AVALIA\\RESULTADOS-{escolha_stemmer}.csv', index=False, sep=';')

    tempo_total = time.time() - start_time
    print("\n\n\n=================================================================================================\n\n")
    print("🎉🎉 Programa executado com sucesso - Trabalho 2 🎉🎉\n")
    print("Tempo total de execução: ", tempo_total, " segundos.")
    print("\n\n=================================================================================================\n\n")

except ValueError as erro:
    print('\n\n\nERROR:', erro)
