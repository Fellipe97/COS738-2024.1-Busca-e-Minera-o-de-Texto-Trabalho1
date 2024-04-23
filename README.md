
    <h1>COS738-2024.1-Busca-e-Minera-o-de-Texto-Trabalho1</h1>

    <p>Trabalho 1 do curso de BMT 2024 01 - COS738 - Busca e Mineração de Texto UFRJ para mestrado</p>

    <h2>Funcionalidades</h2>
    <ul>
        <li><strong>Processador de Consultas:</strong> Lê um arquivo de configuração e um arquivo XML com consultas, processa essas consultas e seus resultados esperados, e os grava em arquivos CSV.</li>
        <li><strong>Gerador de Lista Invertida:</strong> Lê um arquivo de configuração e um conjunto de arquivos XML, extrai resumos de documentos, gera uma lista invertida das palavras nos resumos e a salva em um arquivo CSV.</li>
        <li><strong>Indexador:</strong> Lê um arquivo de configuração e uma lista invertida previamente gerada, calcula pesos para os termos e documentos, e os salva em um arquivo CSV.</li>
        <li><strong>Buscador:</strong> Lê um arquivo de configuração, um modelo vetorial previamente gerado e consultas a serem buscadas, calcula a similaridade entre as consultas e os documentos, e salva os resultados em um arquivo CSV.</li>
    </ul>

    <p>O programa resulta em arquivos CSV com consultas processadas, resultados esperados, lista invertida, pesos de termos e documentos, e resultados da busca, conforme configurado nos arquivos de configuração.</p>

    <h2>Como rodar o programa no seu computador:</h2>
    <ol>
        <li>Crie o ambiente virtual:</li>
        <pre><code>python -m venv venv</code></pre>
        <li>Ative o ambiente virtual:</li>
        <pre><code>.\venv\Scripts\Activate.ps1</code></pre>
        <li>Instale as bibliotecas necessárias:</li>
        <pre><code>pip install -r requirements.txt</code></pre>
        <li>Execute a aplicação estando no ambiente virtual:</li>
        <pre><code>python main.py</code></pre>
    </ol>

    <h2>Qualquer dúvida em como rodar, consulte o arquivo `comandoRodarPrograma.txt`.</h2>

