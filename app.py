import streamlit as st
import os

# Importando o motor que construímos
from src.rag_engine import get_rag_chain

# 1. Configuração visual da página
st.set_page_config(
    page_title="Seller Copilot - Mercado Livre", page_icon="📦", layout="centered"
)

st.title("📦 Copiloto para Vendedores")
st.markdown(
    "Assistente inteligente para suporte a clientes baseado nas suas políticas."
)
st.divider()


# 2. Carregamento em Cache (Otimização crucial)
# O Streamlit recarrega o código todo a cada clique. O @st.cache_resource garante
# que o banco vetorial e os modelos não sejam recarregados do zero toda hora.
@st.cache_resource
def load_chain():
    return get_rag_chain()


try:
    rag_chain = load_chain()
except Exception as e:
    st.error(
        "Erro ao carregar o motor RAG. Verifique sua chave de API e o banco vetorial."
    )
    st.stop()

# 3. Gerenciamento de Memória (Sessão)
# Cria uma lista para guardar o histórico da conversa enquanto a aba estiver aberta
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Adiciona uma mensagem inicial de boas-vindas
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Olá! Eu sou o seu Copiloto de Atendimento. Cole a mensagem do cliente ou faça uma pergunta sobre a política de devoluções.",
        }
    )

# 4. Renderiza o histórico de mensagens na tela
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Caixa de texto para o usuário interagir
prompt = st.chat_input(
    "Ex: Um cliente quer devolver um produto com defeito após 40 dias..."
)

if prompt:
    # Mostra a mensagem do usuário na tela e salva no histórico
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Chama a IA e mostra um "spinner" de carregamento bonito
    with st.chat_message("assistant"):
        with st.spinner("Consultando políticas e redigindo resposta..."):
            try:
                # Dispara a pergunta para o nosso motor RAG
                resposta = rag_chain.invoke(prompt)

                # Exibe a resposta e salva no histórico
                st.markdown(resposta)
                st.session_state.messages.append(
                    {"role": "assistant", "content": resposta}
                )
            except Exception as e:
                st.error(f"Erro na comunicação com a IA: {e}")
