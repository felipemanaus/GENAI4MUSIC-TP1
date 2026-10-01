# One More Morph: Evolutionary MIDI Transitions in Daft Punk Music

Este repositório contém a implementação do Trabalho Prático 1 da disciplina DCC831 - IA Generativa para Música. O projeto utiliza Algoritmos Genéticos para gerar transições graduais (morphing) no domínio simbólico entre duas músicas em formato MIDI (focado no estilo French House / EDM do Daft Punk).

## Dependências e Instalação

Para instalar as bibliotecas necessárias, execute o comando abaixo no seu ambiente virtual:

```bash
pip install -r requirements.txt
```

## Como Reproduzir os Resultados

Certifique-se de que os arquivos MIDI de origem estão na raiz do diretório. Para executar o algoritmo genético e gerar a transição, rode o script principal:

```bash
python codigo_3.py
```

O script irá exportar:
- `transicao_daft_punk.mid`: O arquivo MIDI contendo a transição evolutiva.
- `ga_convergence_curve.png` e `ga_convergence_curve.pdf`: Gráficos científicos documentando a queda da função de perda (fitness) ao longo das gerações.

**Nota sobre os Áudios:** Conforme os requisitos do trabalho, as versões convertidas em áudio sintetizado a partir das saídas MIDI (com variações na taxa de mutação) encontram-se na pasta correspondente do repositório.

## Declaração de Uso de Inteligência Artificial

Conforme as diretrizes da disciplina, ferramentas de Inteligência Artificial (Google Gemini) foram utilizadas como apoio ao desenvolvimento deste projeto. A IA atuou estritamente como um assistente de pesquisa e estruturação, realizando as seguintes tarefas:

- **Levantamento Bibliográfico:** Fornecimento de prompts otimizados para busca de artigos e identificação de trabalhos cruciais para a modelagem do problema (MusicBlox, algoritmo TraSe e o modelo de Komarov et al.).
- **Discussão da Arquitetura Evolutiva:** Apoio na definição da estrutura do Algoritmo Genético, especificamente na decisão de utilizar a trajetória de gerações (concatenando os melhores indivíduos de marcos evolutivos) em vez de focar num único indivíduo final.
- **Refinamento de Código Auxiliar:** Sugestão de funções modulares para a função de fitness (penalização tonal, distância ao alvo e rugosidade horizontal) e operadores de mutação adaptados para trabalhar com blocos matriciais, evitando ruído em vez de mutações aleatórias em células de 100ms.
- **Revisão e Polimento:** Sugestões de formatação visual para os gráficos usando `matplotlib` e revisão gramatical/estrutural do artigo no formato ISMIR.

A escrita final do artigo, a implementação unificada do código, os testes empíricos e as escolhas temáticas foram de autoria inteiramente humana. A IA não foi utilizada para gerar automaticamente a implementação na íntegra nem para redigir o resumo final.

**Link para a conversa com a IA:** https://share.gemini.google/lJ7ay2NE6Jas