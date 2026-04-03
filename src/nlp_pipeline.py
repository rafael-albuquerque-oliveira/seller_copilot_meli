import polars as pl
import google.generativeai as genai
import json
import os
import time

from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Agora o os.environ.get vai encontrar a sua chave perfeitamente
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "A variável GEMINI_API_KEY não foi encontrada. Verifique seu arquivo .env"
    )

genai.configure(api_key=api_key)
# Usando o modelo 'flash' pois é focado em velocidade e tarefas de alta volumetria
model = genai.GenerativeModel("gemini-1.5-flash")


def extrair_metadados_llm(texto: str) -> dict:
    """
    Envia o texto do review para o LLM e retorna um dicionário com sentimento e tópico.
    """
    prompt = f"""
    Atue como um analista de qualidade do Mercado Livre.
    Analise a seguinte avaliação de um cliente: "{texto}"

    Classifique a mensagem e retorne EXATAMENTE um JSON válido com duas chaves:
    - "sentimento": (Positivo, Neutro ou Negativo)
    - "topico": (Escolha a mais adequada entre: Entrega, Qualidade do Produto, Atendimento, Preço ou Outros)

    Não adicione nenhuma formatação Markdown, apenas o objeto JSON puro.
    """

    try:
        response = model.generate_content(prompt)
        # Limpeza defensiva caso o modelo retorne ```json no texto
        texto_limpo = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpo)
    except Exception as e:
        print(f"Erro ao processar texto: {texto[:30]}... Erro: {e}")
        # Retorno padrão em caso de falha (Resiliência)
        return {"sentimento": "N/A", "topico": "N/A"}


def enrich_dataset(input_parquet: str, output_parquet: str) -> None:
    print(f"Carregando dados processados de: {input_parquet}")
    df = pl.read_parquet(input_parquet)

    # Para economizar tempo e requisições no nosso protótipo, vamos pegar apenas os 50 primeiros.
    # Em produção, você processaria o batch inteiro de forma assíncrona.
    df_reduzido = df.head(50)

    textos = df_reduzido["cleaned_message"].to_list()
    resultados = []

    print(f"Iniciando processamento no LLM para {len(textos)} mensagens...")
    for i, texto in enumerate(textos):
        resultado = extrair_metadados_llm(texto)
        resultados.append(resultado)

        # Rate Limiting: Pequena pausa para não estourar o limite de requisições gratuitas da API
        time.sleep(1)
        if (i + 1) % 10 == 0:
            print(f"{i + 1} mensagens processadas...")

    # Separando a lista de dicionários em duas listas para criar as colunas no Polars
    sentimentos = [r.get("sentimento", "N/A") for r in resultados]
    topicos = [r.get("topico", "N/A") for r in resultados]

    # Adicionando as novas colunas estruturadas ao DataFrame
    df_enriquecido = df_reduzido.with_columns(
        [pl.Series("sentimento_llm", sentimentos), pl.Series("topico_llm", topicos)]
    )

    df_enriquecido.write_parquet(output_parquet)
    print(f"Dataset enriquecido salvo em: {output_parquet}")


if __name__ == "__main__":
    INPUT_PATH = "../data/processed/cleaned_reviews.parquet"
    OUTPUT_PATH = "../data/processed/enriched_reviews.parquet"
    enrich_dataset(INPUT_PATH, OUTPUT_PATH)
