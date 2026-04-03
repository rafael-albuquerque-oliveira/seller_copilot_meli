# 📦 Seller Copilot - Mercado Livre (AI/RAG)

O **Seller Copilot** é um assistente inteligente de ponta a ponta desenvolvido para vendedores da plataforma Mercado Livre. Ele utiliza Inteligência Artificial Generativa e a técnica de **RAG (Retrieval-Augmented Generation)** para responder dúvidas de clientes com base nas políticas específicas da loja, reduzindo o tempo de resposta e garantindo conformidade com as regras de negócio.



## 🚀 Diferenciais Técnicos (Por que este projeto?)

Este projeto foi desenhado seguindo as melhores práticas de Engenharia de Machine Learning:

* **Ingestão de Alta Performance:** Utilização do **Polars** em vez do Pandas, garantindo processamento colunar e *lazy evaluation* para lidar com grandes volumes de dados (Dataset Olist).
* **Arquitetura Resiliente:** O sistema utiliza **HuggingFace Embeddings** locais, tornando a etapa de vetorização gratuita e imune a falhas de rede ou limites de taxa (Rate Limits) de APIs externas.
* **Orquestração Moderna (LCEL):** Construído com **LangChain Expression Language**, permitindo um pipeline de dados modular, fácil de depurar e pronto para produção.
* **Inferência de Baixa Latência:** Integração com a infraestrutura do **Groq (Llama 3.1)**, entregando respostas em milissegundos.

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.12+
- **Orquestração de IA:** LangChain (LCEL)
- **Modelos de Linguagem:** Llama 3.1 (via Groq)
- **Embeddings:** HuggingFace (sentence-transformers)
- **Banco Vetorial:** ChromaDB
- **Processamento de Dados:** Polars & PyArrow
- **Interface:** Streamlit

## 📂 Estrutura do Projeto

```text
├── data/
│   ├── knowledge_base/    # Documentos de política (TXT/PDF)
│   └── processed/         # Datasets limpos em formato Parquet
├── src/
│   ├── data_prep.py       # Pipeline de engenharia de dados
│   └── rag_engine.py      # Lógica de Recuperação e IA
├── app.py                 # Interface Web (Streamlit)
└── .env                   # Variáveis de ambiente (Chaves de API)


⚙️ Instalação e Uso
Clonar o repositório:

Bash
git clone [https://github.com/rafael-albuquerque-oliveira/seller_copilot_meli.git](https://github.com/rafael-albuquerque-oliveira/seller_copilot_meli.git)

cd seller_copilot_meli

Configurar Ambiente Virtual:

Bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Configurar Variáveis de Ambiente:
Crie um arquivo .env na raiz e adicione sua chave:

GROQ_API_KEY=gsk_your_key_here

Executar:

Bash
streamlit run app.py

🧠 Insights de Engenharia
Durante o desenvolvimento, o sistema foi "pivotado" de uma solução baseada 100% em nuvem (Gemini) para uma arquitetura híbrida (HuggingFace Local + Groq Cloud). Essa decisão foi tomada para garantir que o retriever (busca no banco de dados) fosse determinístico e gratuito, deixando o custo de tokens apenas para a geração da resposta final.
