# Laboratório 09 - Arquitetura RAG Avançada

Projeto desenvolvido para o Laboratório 09 com foco em técnicas modernas de Retrieval-Augmented Generation (RAG).

---

#  Objetivo

Implementar um pipeline RAG avançado utilizando:

* Embeddings semânticos
* Índice vetorial HNSW
* HyDE (Hypothetical Document Embeddings)
* Re-ranking com Cross-Encoder

O sistema recebe uma consulta em linguagem natural, recupera documentos semanticamente relevantes e reorganiza os resultados utilizando re-ranking neural.

---

#  Tecnologias Utilizadas

* Python
* FAISS
* Sentence Transformers
* Transformers
* Cross-Encoder
* HNSW

---

#  Arquitetura do Projeto

## 1. Embeddings

Os documentos são convertidos em vetores semânticos utilizando:

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

## 2. Índice Vetorial HNSW

Foi utilizado o algoritmo HNSW (Hierarchical Navigable Small World) através do FAISS.

### Parâmetros utilizados

* `M = 32`
* `efConstruction = 100`
* `efSearch = 64`

### Explicação

* `M` controla a quantidade de conexões entre os nós do grafo

* valores maiores aumentam precisão, porém usam mais memória

* `efConstruction` controla a qualidade da construção do índice

* valores maiores tornam a busca mais precisa, porém aumentam o tempo de indexação

---

## 3. HyDE

Foi utilizada a técnica HyDE (Hypothetical Document Embeddings).

A consulta do usuário é transformada em um documento técnico hipotético antes da busca vetorial.

Exemplo:

### Query original

```text
dor de cabeça latejante e luz incomodando
```

### Documento hipotético

```text
Paciente apresenta cefaleia pulsátil associada à fotofobia.
```

Isso melhora a recuperação semântica.

---

## 4. Re-ranking com Cross-Encoder

Após recuperar os Top-10 documentos via HNSW, foi utilizado um Cross-Encoder multilíngue para reorganizar os resultados.

Modelo utilizado:

```text
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

O Cross-Encoder analisa query e documento simultaneamente, produzindo ranking mais preciso.

---

#  Estrutura

```text
lab-09-rag-avancado/
│
├── data/
│   └── manuals.json
│
├── src/
│   ├── build_index.py
│   ├── hyde_query.py
│   └── main.py
│   ├── rerank.py
│   ├── retrieve.py
│
└── .gitignore
├── requirements.txt
├── README.md
```

---

#  Como Executar

## 1. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## 2. Executar o sistema

```bash
python src/main.py
```

---

#  Funcionamento

O pipeline executa:

1. Geração do documento hipotético (HyDE)
2. Criação de embeddings
3. Busca Top-10 via HNSW
4. Re-ranking via Cross-Encoder
5. Retorno do Top-3 final

---

#  Resultado

O sistema conseguiu recuperar corretamente documentos semanticamente relacionados à consulta do usuário, demonstrando funcionamento completo da arquitetura RAG avançada.

---

#  Uso de IA

Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Guilherme Benigno.
