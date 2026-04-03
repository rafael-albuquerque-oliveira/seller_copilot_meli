import os
from dotenv import load_dotenv

# Importações Modernas do LangChain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # <-- Nova biblioteca otimizada para o Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Carrega chaves do .env
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "Configure a variável GEMINI_API_KEY no seu arquivo .env antes de rodar."
    )

CHROMA_PATH = "../chroma_db"
POLICY_PATH = "../data/knowledge_base/politica_devolucao.txt"


def build_vector_store():
    print("Iniciando a leitura do documento de políticas...")
    loader = TextLoader(POLICY_PATH, encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Documento dividido em {len(chunks)} fragmentos.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )

    print("Criando banco vetorial local (ChromaDB)...")
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH
    )
    print(f"Banco vetorial criado com sucesso em: {CHROMA_PATH}")
    return vector_store


def get_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=api_key, temperature=0.2
    )

    system_prompt = (
        "Você é o 'Copiloto de Vendedores' do Mercado Livre. "
        "Use os seguintes trechos da política da loja para responder à pergunta do cliente. "
        "Baseie-se APENAS no contexto fornecido. Se a resposta não estiver no contexto, diga que "
        "não há informações suficientes.\n\n"
        "Contexto recuperado:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


if __name__ == "__main__":
    if not os.path.exists(CHROMA_PATH):
        build_vector_store()
    else:
        print("Banco vetorial já existe, pulando criação...")

    chain = get_rag_chain()

    cenario_cliente = "Comprei um fone de ouvido de vocês, chegou faz 15 dias, mas me arrependi da cor. Quero devolver, como faço?"

    print("\n--- TESTE DO RAG ---")
    print(f"Mensagem do Cliente: '{cenario_cliente}'")
    print("Buscando regras e gerando resposta...\n")

    resposta = chain.invoke({"input": cenario_cliente})
    print(resposta["answer"])
