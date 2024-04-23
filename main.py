import nltk
from xml.etree import ElementTree as ET
import os
import math
import time
import csv
from ast import literal_eval
from collections import defaultdict
from math import sqrt


startTime = time.time()

print("\n\n\n=================================================================================================\n\n")
print("IMPLEMENTACAO DE UM SISTEMA DE RECUPERACAO EM MEMORIA SEGUNDO O MODELO VETORIAL")
print("\n\n=================================================================================================\n\n")

# Processador de Consultas
print("\n=================================================================================================\n")
print("1 - PROCESSADOR DE CONSULTAS \n")

print("❗Lendo o arquivo de configuracao config/PC.CFG...")
configuracao_pc = {}
caminho_arquivo_pc = os.getcwd() + '/config/PC.CFG'
with open(caminho_arquivo_pc, 'r') as arquivo:
    for linha in arquivo:
        chave, valor = linha.strip().split('=')
        configuracao_pc[chave.strip()] = valor.strip()

arquivo_xml_pc = configuracao_pc["LEIA"]
instrucao_consulta_pc = configuracao_pc["CONSULTAS"]
instrucao_esperada_pc = configuracao_pc["ESPERADOS"]

consultas_pc = {}

arvore_pc = ET.parse(os.getcwd() + '/data/' + arquivo_xml_pc)
raiz_pc = arvore_pc.getroot()

print("\n❗Processando as consultas...")
for consulta_bruta in raiz_pc.iter('QUERY'):
    numero_consulta = None
    texto_consulta = None
    resultados_consulta = {}

    for elemento in consulta_bruta.iter():
        if elemento.tag == 'QueryNumber':
            numero_consulta = elemento.text.strip()
        elif elemento.tag == 'QueryText':
            texto_consulta = elemento.text.strip().upper()
        elif elemento.tag == 'Records':
            for item in elemento.iter('Item'):
                numero_documento = int(item.text.strip())
                pontuacao_documento = int(item.attrib.get('score'))
                resultados_consulta[numero_documento] = sum(int(digito) for digito in str(pontuacao_documento))

    if numero_consulta and texto_consulta:
        consultas_pc[int(numero_consulta)] = {'texto': texto_consulta, 'resultados': resultados_consulta}

print("\n❗Gravando as consultas processadas no arquivo CSV...")
with open(os.getcwd() + '/RESULT/' + instrucao_consulta_pc, 'w') as arquivo_consultas_processadas:
    arquivo_consultas_processadas.write('QueryNumber;QueryText\n')
    for numero_consulta, dados_consulta in consultas_pc.items():
        texto_consulta = dados_consulta['texto'].replace('\n', ' ').strip().replace("    ", " ")
        arquivo_consultas_processadas.write(f'{numero_consulta};"{texto_consulta}"\n')

print("\n❗Gravando os resultados esperados da consulta..")
with open(os.getcwd() + '/RESULT/' + instrucao_esperada_pc, 'w') as arquivo_resultados_esperados:
    arquivo_resultados_esperados.write('QueryNumber;DocNumber;DocScore\n')
    for numero_consulta, dados_consulta in consultas_pc.items():
        for numero_documento, pontuacao_documento in dados_consulta['resultados'].items():
            arquivo_resultados_esperados.write(f'{numero_consulta};{numero_documento};{pontuacao_documento}\n')

fimConsulta = time.time() - startTime
print('\n\n🙌 Arquivos gerados na pasta RESULT: ', instrucao_consulta_pc, " e ", instrucao_esperada_pc)
print("\n🕑 Tempo: ", fimConsulta)
print("\n✅ Encerramento do processador de consultas.\n")
print("=================================================================================================\n\n")











# Gerador de Lista Invertida
print("\n=================================================================================================\n")
print("2 - GERADOR LISTA INVERTIDA\n")
inicio_gerador_lista_invertida = time.time()

print("❗Lendo o arquivo de configuracao config/GLI.CFG...")

config_gli = {}
caminho_arquivo_gli = os.getcwd() + '/config/GLI.CFG'
with open(caminho_arquivo_gli, 'r') as arquivo:
    for linha in arquivo:
        chave, valor = linha.strip().split('=')
        config_gli[chave.strip()] = valor.strip()

arquivos_xml_gli = config_gli["LEIA"].split(', ')
nome_lista_invertida_gli = config_gli["ESCREVA"]

print('\n❗Processando os arquivos...')
documentos_gli = {}
for arquivo_xml in [os.path.join(os.getcwd(), arquivo) for arquivo in arquivos_xml_gli]:
    arvore = ET.parse(arquivo_xml)
    raiz = arvore.getroot()
    for registro in raiz.findall('RECORD'):
        numero_registro = int(registro.find('RECORDNUM').text)
        resumo = registro.find('ABSTRACT')
        extrato = registro.find('EXTRACT')
        texto_resumo = resumo.text if resumo is not None else extrato.text if extrato is not None else ""
        documentos_gli[numero_registro] = nltk.tokenize.word_tokenize(texto_resumo)

print('\n❗Carregando os stopwords...')
stop_words_gli = ''
with open(os.getcwd() + '/stopwords/stopwords.txt', 'r') as arquivo:
    stop_words_gli = [palavra.strip().upper() for palavra in arquivo.readlines()]

print('\n❗Gerando a lista invertida...')
lista_invertida_gli = {}
for numero_registro, texto_resumo in documentos_gli.items():
    if isinstance(texto_resumo, list):
        texto_resumo = ' '.join(texto_resumo)
    palavras = texto_resumo.split()
    for palavra in palavras:
        palavra = palavra.upper().strip(';')
        if palavra not in stop_words_gli:
            if palavra not in lista_invertida_gli:
                lista_invertida_gli[palavra] = [numero_registro]
            else:
                lista_invertida_gli[palavra].append(numero_registro)

print('\n❗ Gravando a lista invertida...')  
with open(os.getcwd() + nome_lista_invertida_gli, 'w') as arquivo_csv:
    arquivo_csv.write('Palavra;DocumentIDs\n')
    for palavra, ids_doc in lista_invertida_gli.items():
        arquivo_csv.write(f'{palavra};{ids_doc}\n')

fim_gerador_lista_invertida = time.time() - inicio_gerador_lista_invertida
print('\n\n🙌 Arquivos gerados na pasta RESULT: ', nome_lista_invertida_gli)
print("\n🕑 Tempo: ", fim_gerador_lista_invertida)
print("\n✅ Encerramento do gerador de lista invertida.\n")
print("=================================================================================================\n\n")











# Indexador
print("\n=================================================================================================\n")
print("3 - INDEXADOR\n")
inicio_indexador = time.time()

print("❗ Lendo o arquivo de configuracao config/INDEX.CFG...")
instrucoes_indexador = {}
with open(os.getcwd() + '/config/INDEX.CFG', 'r') as arquivo:
    for linha in arquivo:
        chave, valor = linha.strip().split('=')
        instrucoes_indexador[chave.strip()] = valor.strip()

lista_invertida_arquivo_indexador = instrucoes_indexador['LEIA']
arquivo_saida_indexador = instrucoes_indexador['ESCREVA']

print("\n❗ Carregando Lista Invertida - Leitura do CSV...")
lista_invertida = {}
with open(os.getcwd() + lista_invertida_arquivo_indexador, 'r') as arquivo:
    leitor = csv.reader(arquivo, delimiter=';')
    next(leitor)
    for linha in leitor:
        palavra = linha[0]
        ids_doc = literal_eval(linha[1])
        lista_invertida[palavra] = ids_doc


print("\n❗ Indexando a Lista Invertida...")
modelo_vetorial = {}
N = len(lista_invertida)

lista_documentos = list(set(id_doc for freq_termos in lista_invertida.values() for id_doc in freq_termos))

for termo, freq_termos in lista_invertida.items():
    documentos_ocorreram = list(set(freq_termos))
    idf = math.log(len(lista_documentos) / len(documentos_ocorreram))
    dados_documento = {}

    for documento in documentos_ocorreram:            
        if documento in lista_invertida[termo]:
            tf = lista_invertida[termo].count(documento)
            tf_n = tf / len(lista_invertida[termo])
        else:
            tf_n = 0
    
        peso = tf_n * idf
        dados_documento[documento] = peso

    modelo_vetorial[termo] = (idf, dados_documento)


print("\n❗ Gravando o vetor modelo...")
campos = ['palavra', 'dados']
with open(os.getcwd() + arquivo_saida_indexador, 'w+') as arquivo_csv:
    escritor = csv.DictWriter(arquivo_csv, delimiter=';', lineterminator='\n', fieldnames=campos)
    escritor.writeheader()

    for termo, (idf, dados_documento) in modelo_vetorial.items():
        escritor.writerow({'palavra': termo, 'dados': dados_documento})

fim_indexador = time.time() - inicio_indexador
print('\n\n🙌 Arquivos gerados na pasta RESULT: ', arquivo_saida_indexador)
print("\n🕑 Tempo: ", fim_indexador)
print("\n✅ Encerramento do indexador.\n")
print("=================================================================================================\n\n")











# BUSCADOR
print("\n=================================================================================================\n")
print("4 - BUSCADOR\n\n")
inicio_busca = time.time()

print("❗Lendo o arquivo de configuracao config/BUSCA.CFG...")
instrucoes_indexador = {}
with open(os.getcwd() + '/config/BUSCA.CFG', 'r') as arquivo:
    for linha in arquivo:
        chave, valor = linha.strip().split('=')
        instrucoes_indexador[chave.strip()] = valor.strip()


modelo_vetorial_arquivo = instrucoes_indexador['MODELO']
consultas_arquivo = instrucoes_indexador['CONSULTAS']
resultados_arquivo = instrucoes_indexador['RESULTADOS']


print("\n❗ Carregando modelo vetorial...")
modelo_vetorial = defaultdict(list)
with open(os.getcwd()+modelo_vetorial_arquivo, 'r') as arquivo:
    next(arquivo) 
    for linha in arquivo:
        palavra, dados_str = linha.strip().split(';')
        dados = literal_eval(dados_str)
        for doc_id, peso in dados.items():
            modelo_vetorial[doc_id].append((palavra, peso))


print("\n❗ Carregando as consultas...")
consultas = defaultdict(list)
with open(os.getcwd()+consultas_arquivo, 'r') as arquivo:
    linhas = arquivo.readlines()
    i = 1  
    while i < len(linhas):
        linha = linhas[i].strip()  
        if linha:  
            while i + 1 < len(linhas) and linhas[i + 1].startswith(' '):
                linha += linhas[i + 1].lstrip()
                i += 1     
            partes = linha.split(';', 1)
            if len(partes) == 2:
                numero_consulta = int(partes[0].strip())
                texto_consulta = partes[1].strip().strip('"').split()
                
                consultas[numero_consulta] = texto_consulta
            else:
                print(f"Erro na linha: {linha}")
        i += 1  

print("\n❗ Buscando a resposta para a consulta...")
resultados_busca = []
for i, consulta in enumerate(consultas, start=1):
    resultados_consulta = []
    for doc_id, vetor_doc in modelo_vetorial.items():
        vetor_doc = dict(vetor_doc)
        if isinstance(vetor_doc, dict):
            produto_ponto = sum((1 if palavra in consultas[consulta] else 0) * peso for palavra, peso in vetor_doc.items())
            magnitude_consulta = sqrt(len(consultas[consulta]))  
            magnitude_doc = sqrt(sum(peso ** 2 for palavra, peso in vetor_doc.items()))
            if magnitude_consulta == 0 or magnitude_doc == 0:
                similaridade = 0
            similaridade = produto_ponto / (magnitude_consulta * magnitude_doc)
        else:
            similaridade = 0 
        resultados_consulta.append((similaridade, doc_id))
    resultados_consulta.sort(reverse=True) 
    resultados_busca.append((i, resultados_consulta))


print("\n❗ Gravando os resultados...")
with open(os.getcwd()+resultados_arquivo, 'w', newline='') as arquivo_csv:
        escritor_csv = csv.writer(arquivo_csv, delimiter=';')
        for id_consulta, resultados_consulta in resultados_busca:
            for posicao, (similaridade, doc_id) in enumerate(resultados_consulta, start=1):
                escritor_csv.writerow([id_consulta, [posicao, doc_id, similaridade]])


fim_busca = time.time() - inicio_busca
print('\n\n🙌 Arquivos gerados na pasta RESULT: ', resultados_arquivo)
print("\n🕑 Tempo: ", fim_busca)
print("\n✅ Encerramento do buscador.\n")
print("=================================================================================================\n\n")









total_time = time.time() - startTime
print("\n\n\n=================================================================================================\n\n")
print("🎉🎉 Programa executado com sucesso 🎉🎉\n")
print("Tempo ao executar o programa por completo: ", total_time, " segundos.")
print("\n\n=================================================================================================\n\n")