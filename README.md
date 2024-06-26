# COS738-2024.1-Busca-e-Minera-o-de-Texto




## Trabalho1
Trabalho 1 do curso de BMT 2024 01 - COS738 - Busca e Mineração de Texto UFRJ para mestrado

## Funcionalidades
- **Processador de Consultas:** Lê um arquivo de configuração e um arquivo XML com consultas, processa essas consultas e seus resultados esperados, e os grava em arquivos CSV.
- **Gerador de Lista Invertida:** Lê um arquivo de configuração e um conjunto de arquivos XML, extrai resumos de documentos, gera uma lista invertida das palavras nos resumos e a salva em um arquivo CSV.
- **Indexador:** Lê um arquivo de configuração e uma lista invertida previamente gerada, calcula pesos para os termos e documentos, e os salva em um arquivo CSV.
- **Buscador:** Lê um arquivo de configuração, um modelo vetorial previamente gerado e consultas a serem buscadas, calcula a similaridade entre as consultas e os documentos, e salva os resultados em um arquivo CSV.

O programa resulta em arquivos CSV com consultas processadas, resultados esperados, lista invertida, pesos de termos e documentos, e resultados da busca, conforme configurado nos arquivos de configuração.




## Trabalho2
Trabalho 2 expande o projeto inicial, introduzindo componentes analíticos e de avaliação para o sistema de recuperação de informações:

## Funcionalidades
- **Configuração e Preparação de Dados:** Configura diretórios e carrega dados necessários para análise, com opções para utilizar ou não o stemmer, conforme definido em um arquivo de configuração.
- **Pré-processamento e Organização de Dados:** Aplica pré-processamento aos resultados das consultas para padronizar os dados antes da análise.
- **Avaliação de Métricas de Desempenho:** Calcula várias métricas, como F1 Score, Precisão, Recall, e MAP, usando dados organizados para avaliar a eficácia do sistema.
- **Geração de Relatórios e Visualizações:** Produz relatórios e gráficos (como a curva Precision-Recall) para visualizar o desempenho do sistema de forma mais intuitiva.

## Como rodar o programa no seu computador:
1. Crie o ambiente virtual:
    ```bash
    python -m venv venv
    ```
2. Ative o ambiente virtual:
    ```bash
    .\venv\Scripts\Activate.ps1
    ```
3. Instale as bibliotecas necessárias:
    ```bash
    pip install -r requirements.txt
    ```
4. Execute a aplicação estando no ambiente virtual:

    4.1 Para rodar o trabalho 1:
    ```bash
    python main.py
    ```
    4.2 Para rodar o trabalho 2:
    ```bash
    python main2.py
    ```

## Qualquer dúvida em como rodar, consulte o arquivo `comandoRodarPrograma.txt`.
