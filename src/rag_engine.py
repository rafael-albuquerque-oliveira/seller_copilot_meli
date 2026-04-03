import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings  # <--- A Solução Open Source
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "Configure a variável GEMINI_API_KEY no seu arquivo .env antes de rodar."
    )

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
POLICY_PATH = os.path.join(BASE_DIR, "data", "knowledge_base", "politica_devolucao.txt")


# Função que retorna os embeddings locais (Rápido, gratuito e imune a erros 404 de API)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vector_store():
    print("Iniciando a leitura do documento de políticas...")
    loader = TextLoader(POLICY_PATH, encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Documento dividido em {len(chunks)} fragmentos.")

    print("Criando banco vetorial local (ChromaDB) com HuggingFace Embeddings...")
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=get_embeddings(), persist_directory=CHROMA_PATH
    )
    print(f"Banco vetorial criado com sucesso em: {CHROMA_PATH}")
    return vector_store


def get_rag_chain():
    vector_store = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embeddings()
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash", google_api_key=api_key, temperature=0.2
    # )
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

    system_prompt = (
        "Você é o 'Copiloto de Vendedores' do Mercado Livre. "
        "Use os seguintes trechos da política da loja para responder à pergunta do cliente. "
        "INSTRUÇÕES DE RACIOCÍNIO: "
        "- Faça deduções lógicas simples (exemplo: 'fone de ouvido' faz parte da categoria 'Eletrônicos'; 'não gostei da cor' é o mesmo que devolução por 'arrependimento'). "
        "- Seja educado, direto e aja como o vendedor resolvendo o problema do cliente. "
        "- Baseie-se nas regras de prazo e frete do contexto. Se uma regra não estiver lá, diga que não tem informações suficientes.\n\n"
        "Contexto recuperado:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":

    import shutil

    if os.path.exists(CHROMA_PATH):
        print("Limpando banco vetorial antigo...")
        shutil.rmtree(CHROMA_PATH)

    build_vector_store()

    chain = get_rag_chain()

    cenario_cliente = "Comprei um fone de ouvido de vocês, chegou faz 15 dias, mas me arrependi da cor. Quero devolver, como faço?"

    print("\n--- TESTE DO RAG ---")
    print(f"Mensagem do Cliente: '{cenario_cliente}'")
    print("Buscando regras e gerando resposta...\n")

    resposta = chain.invoke(cenario_cliente)
    print(resposta)
